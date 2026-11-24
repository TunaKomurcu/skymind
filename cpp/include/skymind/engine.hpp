#pragma once

#include <string>

namespace skymind {

struct GenerationResult {
  std::string text;
  int prompt_tokens = 0;
  int new_tokens = 0;
  double prefill_ms = 0;   // istem işleme
  double decode_ms = 0;    // token üretimi
  bool truncated = false;  // max_new_tokens'a takıldı
};

// Çıkarım motoru soyutlaması: llama.cpp (bu repoda), ileride ONNX Runtime. Çağrılar iş parçacığı
// güvenli olmak zorunda; tek bağlamlı motorlar kendi içinde sıraya sokar.
class InferenceEngine {
 public:
  virtual ~InferenceEngine() = default;
  virtual GenerationResult generate(const std::string& system, const std::string& command) = 0;
  virtual std::string name() const = 0;
};

}  // namespace skymind
