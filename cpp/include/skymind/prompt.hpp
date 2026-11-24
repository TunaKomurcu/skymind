#pragma once

#include <string>

namespace skymind {

// Qwen2.5 sohbet biçimi (ChatML), eğitimdeki şablonla aynı: sistem + kullanıcı + asistan başlangıcı.
std::string chatml_prompt(const std::string& system, const std::string& user);

// Dosyadan sistem istemi okur (sondaki satır sonlarını kırpar). Dosya yoksa std::runtime_error.
std::string read_system_prompt(const std::string& path);

}  // namespace skymind
