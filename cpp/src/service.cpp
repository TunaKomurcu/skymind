#include "skymind/service.hpp"

#include <chrono>

#include "skymind/validate.hpp"

namespace skymind {
namespace {

PlanResponse error_response(int status, const std::string& message) {
  return {status, {{"error", message}}};
}

}  // namespace

PlannerService::PlannerService(std::shared_ptr<InferenceEngine> engine, std::string system_prompt,
                               Metrics& metrics)
    : engine_(std::move(engine)), system_(std::move(system_prompt)), metrics_(metrics) {
  metrics_.describe("skymind_requests_total", "counter", "Plan istekleri (durum etiketiyle)");
  metrics_.describe("skymind_output_total", "counter", "Uretilen cikti turleri (gecersiz dahil)");
  metrics_.describe("skymind_prompt_tokens_total", "counter", "Islenen istem token sayisi");
  metrics_.describe("skymind_generated_tokens_total", "counter", "Uretilen token sayisi");
  metrics_.describe("skymind_requests_in_flight", "gauge", "Islenmekte olan istek sayisi");
  metrics_.describe("skymind_request_duration_seconds", "histogram", "Uctan uca istek suresi");
  metrics_.set_gauge("skymind_requests_in_flight", 0);
}

PlanResponse PlannerService::plan(const std::string& body) {
  const auto t0 = std::chrono::steady_clock::now();
  nlohmann::json req = nlohmann::json::parse(body, nullptr, false);
  if (req.is_discarded() || !req.is_object() || !req.contains("command") ||
      !req["command"].is_string()) {
    metrics_.inc("skymind_requests_total", "status=\"bad_request\"");
    return error_response(400, "govde {\"command\": \"...\"} bicimde JSON olmali");
  }
  const std::string command = req["command"].get<std::string>();
  if (command.empty() || command.size() > kMaxCommandBytes) {
    metrics_.inc("skymind_requests_total", "status=\"bad_request\"");
    return error_response(400, "command bos olamaz ve en fazla 2000 bayt olabilir");
  }

  metrics_.add_gauge("skymind_requests_in_flight", 1);
  PlanResponse res;
  try {
    const GenerationResult g = engine_->generate(system_, command);
    const Validation v = validate_text(g.text);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    metrics_.inc("skymind_requests_total", "status=\"ok\"");
    metrics_.inc("skymind_output_total",
                 v.ok ? "type=\"" + v.type + "\"" : std::string("type=\"invalid\""));
    metrics_.inc("skymind_prompt_tokens_total", "", g.prompt_tokens);
    metrics_.inc("skymind_generated_tokens_total", "", g.new_tokens);
    metrics_.observe("skymind_request_duration_seconds", secs);

    nlohmann::json out;
    out["raw"] = g.text;
    out["valid"] = v.ok;
    out["type"] = v.ok ? nlohmann::json(v.type) : nlohmann::json(nullptr);
    out["validation_error"] = v.ok ? nlohmann::json(nullptr) : nlohmann::json(v.error);
    out["plan"] = v.ok ? nlohmann::json::parse(g.text) : nlohmann::json(nullptr);
    out["usage"] = {{"prompt_tokens", g.prompt_tokens}, {"new_tokens", g.new_tokens},
                    {"prefill_ms", g.prefill_ms},       {"decode_ms", g.decode_ms},
                    {"truncated", g.truncated}};
    out["latency_ms"] = secs * 1000.0;
    out["engine"] = engine_->name();
    res = {200, std::move(out)};
  } catch (const std::exception& e) {
    metrics_.inc("skymind_requests_total", "status=\"error\"");
    res = error_response(500, std::string("cikarim hatasi: ") + e.what());
  }
  metrics_.add_gauge("skymind_requests_in_flight", -1);
  return res;
}

}  // namespace skymind
