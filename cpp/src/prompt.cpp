#include "skymind/prompt.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace skymind {

std::string chatml_prompt(const std::string& system, const std::string& user) {
  return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user +
         "<|im_end|>\n<|im_start|>assistant\n";
}

std::string read_system_prompt(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("sistem istemi dosyası açılamadı: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}

}  // namespace skymind
