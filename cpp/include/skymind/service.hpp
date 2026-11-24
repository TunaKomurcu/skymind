#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "skymind/engine.hpp"
#include "skymind/metrics.hpp"

namespace skymind {

struct PlanResponse {
  int http_status = 200;
  nlohmann::json body;
};

// HTTP'den bağımsız iş mantığı: istek doğrulama, çıkarım, şema doğrulama, metrikler.
class PlannerService {
 public:
  PlannerService(std::shared_ptr<InferenceEngine> engine, std::string system_prompt,
                 Metrics& metrics);

  // body: {"command": "..."} JSON metni
  PlanResponse plan(const std::string& body);

  static constexpr size_t kMaxCommandBytes = 2000;

 private:
  std::shared_ptr<InferenceEngine> engine_;
  std::string system_;
  Metrics& metrics_;
};

}  // namespace skymind
