#include "skymind/llama_engine.hpp"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

#include "llama.h"
#include "skymind/prompt.hpp"

namespace skymind {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void silent_log(enum ggml_log_level, const char*, void*) {}

// Verilen token'ları (başlangıç konumundan) tek grup olarak hazırlar; yalnızca sonuncunun logit'i istenir.
void set_tokens(llama_batch_ext* batch, const llama_token* tokens, int32_t n, llama_pos pos0) {
  llama_batch_ext_clear(batch);
  for (int32_t i = 0; i < n; ++i) {
    const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
    const llama_pos pos = pos0 + i;
    llama_batch_ext_set_pos(batch, idx, &pos);
  }
  llama_batch_ext_set_output_logits(batch, n - 1, true);
}

}  // namespace

LlamaEngine::LlamaEngine(const LlamaOptions& opts) : opts_(opts) {
  if (opts_.quiet) llama_log_set(silent_log, nullptr);
  llama_backend_init();

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = opts_.n_gpu_layers;
  model_ = llama_model_load_from_file(opts_.model_path.c_str(), mp);
  if (!model_) throw std::runtime_error("model yuklenemedi: " + opts_.model_path);
  vocab_ = llama_model_get_vocab(model_);

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = static_cast<uint32_t>(opts_.n_ctx);
  cp.n_batch = static_cast<uint32_t>(opts_.n_ctx);  // istem tek seferde işlensin
  const int threads =
      opts_.n_threads > 0 ? opts_.n_threads : static_cast<int>(std::thread::hardware_concurrency());
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.no_perf = true;
  ctx_ = llama_init_from_model(model_, cp);
  if (!ctx_) {
    llama_model_free(model_);
    model_ = nullptr;
    throw std::runtime_error("llama baglami olusturulamadi");
  }
  name_ = "llama.cpp:" + std::filesystem::path(opts_.model_path).filename().string();
}

LlamaEngine::~LlamaEngine() {
  if (ctx_) llama_free(ctx_);
  if (model_) llama_model_free(model_);
  llama_backend_free();
}

GenerationResult LlamaEngine::generate(const std::string& system, const std::string& command) {
  std::lock_guard<std::mutex> lock(mu_);
  GenerationResult out;

  const std::string prompt = chatml_prompt(system, command);
  // ilk çağrı boyutu verir (negatif döner), ikincisi doldurur
  const int32_t need = -llama_tokenize(vocab_, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                       nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
  if (need <= 0) throw std::runtime_error("istem tokenlastirilamadi");
  std::vector<llama_token> tokens(static_cast<size_t>(need));
  if (llama_tokenize(vocab_, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(),
                     need, true, true) < 0)
    throw std::runtime_error("istem tokenlastirilamadi");
  out.prompt_tokens = need;

  if (need >= opts_.n_ctx) throw std::runtime_error("istem baglam penceresine sigmiyor");
  const int budget = std::min(opts_.max_new_tokens, opts_.n_ctx - need);

  llama_memory_clear(llama_get_memory(ctx_), true);  // istekler bağımsız

  llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  llama_batch_ext* batch = llama_batch_ext_init(ctx_);

  try {
    set_tokens(batch, tokens.data(), need, 0);
    llama_pos pos = need;
    auto t_prefill = Clock::now();
    bool first = true;

    while (out.new_tokens < budget) {
      if (llama_process(ctx_, LLAMA_PROCESS_TYPE_DECODE, batch) != 0)
        throw std::runtime_error("llama_process basarisiz");
      if (first) {
        out.prefill_ms = ms_since(t_prefill);
        first = false;
      }
      const llama_token tok = llama_sampler_sample(smpl, ctx_, -1);
      if (llama_vocab_is_eog(vocab_, tok)) break;

      char buf[256];
      const int n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf), 0, /*special=*/false);
      if (n < 0) throw std::runtime_error("token metne cevrilemedi");
      out.text.append(buf, static_cast<size_t>(n));
      ++out.new_tokens;

      set_tokens(batch, &tok, 1, pos++);
    }
    out.truncated = out.new_tokens >= budget;
    out.decode_ms = ms_since(t_prefill) - out.prefill_ms;
  } catch (...) {
    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    throw;
  }
  llama_batch_ext_free(batch);
  llama_sampler_free(smpl);
  return out;
}

}  // namespace skymind
