#pragma once

#include <mutex>
#include <string>

#include "skymind/engine.hpp"

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace skymind {

struct LlamaOptions {
  std::string model_path;
  int n_gpu_layers = 0;  // GGML_CUDA kapalı derlemede etkisiz
  int n_ctx = 2048;
  int n_threads = 0;  // 0 = donanım iş parçacığı sayısı
  int max_new_tokens = 400;
  bool quiet = true;  // llama.cpp günlüklerini sustur
};

// llama.cpp tabanlı motor. Tek bağlam (batch=1): istekler kilitle sıraya girer, her istekte KV önbelleği
// temizlenir (istekler birbirinden bağımsız). Açgözlü (greedy) örnekleme, Python değerlendirmesiyle aynı.
class LlamaEngine : public InferenceEngine {
 public:
  explicit LlamaEngine(const LlamaOptions& opts);
  ~LlamaEngine() override;
  LlamaEngine(const LlamaEngine&) = delete;
  LlamaEngine& operator=(const LlamaEngine&) = delete;

  GenerationResult generate(const std::string& system, const std::string& command) override;
  std::string name() const override { return name_; }

 private:
  LlamaOptions opts_;
  std::string name_;
  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  std::mutex mu_;
};

}  // namespace skymind
