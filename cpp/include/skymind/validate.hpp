// Model çıktısının görev planı şemasına uygunluğunu denetler.
//
// skymind/schemas/plan.py (Pydantic) ile aynı kuralları uygular: bilinmeyen alan yasak, tip ayırıcı
// zorunlu, sayısal sınırlar. Eşlik testi: cpp/tests/test_parity.cpp, Python'dan üretilmiş
// geçerli/geçersiz örneklerde iki gerçeklemenin aynı kararı verdiğini doğrular.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace skymind {

// Şema sabitleri (plan.py ile aynı).
constexpr double kMinAltM = 10.0, kMaxAltM = 5000.0;
constexpr double kMinRadiusM = 50.0, kMaxRadiusM = 50000.0;
constexpr double kMinSpeedMs = 5.0, kMaxSpeedMs = 100.0;
constexpr int kMaxSteps = 20;

struct Validation {
  bool ok = false;
  std::string type;   // "plan" | "clarification" | "refusal" (geçerliyse)
  std::string error;  // geçersizse ilk hata, "steps[2].altitude_m: ..." biçiminde
};

Validation validate_output(const nlohmann::json& doc);

// Metni ayrıştırıp doğrular; ayrıştırma hatası da geçersiz sayılır.
Validation validate_text(const std::string& text);

}  // namespace skymind
