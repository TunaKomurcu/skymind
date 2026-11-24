#include "skymind/validate.hpp"

#include <cctype>
#include <cmath>
#include <initializer_list>
#include <set>
#include <stdexcept>

namespace skymind {
namespace {

using nlohmann::json;

struct SchemaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
  throw SchemaError(path + ": " + msg);
}

std::string join(const std::string& base, const std::string& key) {
  return base.empty() ? key : base + "." + key;
}

const json& require_object(const json& j, const std::string& path) {
  if (!j.is_object()) fail(path.empty() ? "<root>" : path, "nesne bekleniyordu");
  return j;
}

// Zorunlu ve isteğe bağlı alanların dışında anahtar yasak (Pydantic extra="forbid").
void check_keys(const json& obj, const std::string& path, std::initializer_list<const char*> required,
                std::initializer_list<const char*> optional) {
  std::set<std::string> allowed;
  for (const char* k : required) {
    allowed.insert(k);
    if (!obj.contains(k)) fail(join(path, k), "zorunlu alan eksik");
  }
  for (const char* k : optional) allowed.insert(k);
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    if (!allowed.count(it.key())) fail(join(path, it.key()), "bilinmeyen alan");
  }
}

bool is_number(const json& j) { return j.is_number(); }  // bool sayılmaz (nlohmann ayırır)

double number_in(const json& obj, const std::string& path, const char* key, double lo, double hi,
                 bool exclusive_lo = false) {
  const json& v = obj.at(key);
  const std::string p = join(path, key);
  if (!is_number(v)) fail(p, "sayı bekleniyordu");
  const double x = v.get<double>();
  if (!std::isfinite(x)) fail(p, "sonlu sayı bekleniyordu");
  if (exclusive_lo ? x <= lo : x < lo) fail(p, "alt sınırın altında");
  if (x > hi) fail(p, "üst sınırın üstünde");
  return x;
}

int integer_in(const json& obj, const std::string& path, const char* key, int lo, int hi) {
  const json& v = obj.at(key);
  const std::string p = join(path, key);
  if (!is_number(v)) fail(p, "tamsayı bekleniyordu");
  const double x = v.get<double>();
  if (x != std::floor(x)) fail(p, "tamsayı bekleniyordu");  // 3.0 kabul, 3.5 red (Pydantic gibi)
  if (x < lo || x > hi) fail(p, "sınır dışı");
  return static_cast<int>(x);
}

// UTF-8 kod noktası sayısı (Pydantic uzunluğu karakter olarak sayar).
size_t code_points(const std::string& s) {
  size_t n = 0;
  for (unsigned char c : s) {
    if ((c & 0xC0) != 0x80) ++n;
  }
  return n;
}

std::string string_len(const json& v, const std::string& path, size_t lo, size_t hi) {
  if (!v.is_string()) fail(path, "metin bekleniyordu");
  const auto s = v.get<std::string>();
  const size_t n = code_points(s);
  if (n < lo || n > hi) fail(path, "uzunluk sınır dışı");
  return s;
}

bool blank(const std::string& s) {
  for (unsigned char c : s) {
    if (!std::isspace(c)) return false;
  }
  return true;
}

void location(const json& j, const std::string& path) {
  require_object(j, path);
  if (!j.contains("kind") || !j["kind"].is_string()) fail(join(path, "kind"), "ayırıcı eksik");
  const auto kind = j["kind"].get<std::string>();
  if (kind == "coords") {
    check_keys(j, path, {"kind", "lat", "lon"}, {});
    number_in(j, path, "lat", -90, 90);
    number_in(j, path, "lon", -180, 180);
  } else if (kind == "named") {
    check_keys(j, path, {"kind", "name"}, {});
    string_len(j["name"], join(path, "name"), 1, 80);
  } else if (kind == "relative") {
    check_keys(j, path, {"kind", "direction"}, {"distance_m", "origin"});
    static const std::set<std::string> dirs = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
    if (!j["direction"].is_string() || !dirs.count(j["direction"].get<std::string>()))
      fail(join(path, "direction"), "geçersiz yön");
    if (j.contains("distance_m") && !j["distance_m"].is_null())
      number_in(j, path, "distance_m", 0, kMaxRadiusM, /*exclusive_lo=*/true);
    if (j.contains("origin")) {
      const auto& o = j["origin"];
      if (!o.is_string() || (o != "home" && o != "current")) fail(join(path, "origin"), "geçersiz");
    }
  } else {
    fail(join(path, "kind"), "bilinmeyen konum türü");
  }
}

bool present(const json& obj, const char* key) { return obj.contains(key) && !obj[key].is_null(); }

void action(const json& j, const std::string& path) {
  require_object(j, path);
  if (!j.contains("action") || !j["action"].is_string()) fail(join(path, "action"), "ayırıcı eksik");
  const auto a = j["action"].get<std::string>();
  if (a == "takeoff") {
    check_keys(j, path, {"action", "altitude_m"}, {});
    number_in(j, path, "altitude_m", kMinAltM, kMaxAltM);
  } else if (a == "goto") {
    check_keys(j, path, {"action", "target"}, {"altitude_m", "speed_ms"});
    location(j["target"], join(path, "target"));
    if (present(j, "altitude_m")) number_in(j, path, "altitude_m", kMinAltM, kMaxAltM);
    if (present(j, "speed_ms")) number_in(j, path, "speed_ms", kMinSpeedMs, kMaxSpeedMs);
  } else if (a == "scan") {
    check_keys(j, path, {"action", "center", "radius_m", "altitude_m"}, {});
    location(j["center"], join(path, "center"));
    number_in(j, path, "radius_m", kMinRadiusM, kMaxRadiusM);
    number_in(j, path, "altitude_m", kMinAltM, kMaxAltM);
  } else if (a == "patrol") {
    check_keys(j, path, {"action", "waypoints", "altitude_m"}, {"loops"});
    const auto& w = j["waypoints"];
    if (!w.is_array() || w.size() < 2 || w.size() > 10)
      fail(join(path, "waypoints"), "2-10 öğeli liste bekleniyordu");
    for (size_t i = 0; i < w.size(); ++i)
      location(w[i], join(path, "waypoints[" + std::to_string(i) + "]"));
    number_in(j, path, "altitude_m", kMinAltM, kMaxAltM);
    if (j.contains("loops")) integer_in(j, path, "loops", 1, 20);
  } else if (a == "loiter") {
    check_keys(j, path, {"action", "center", "altitude_m"}, {"radius_m", "duration_s"});
    location(j["center"], join(path, "center"));
    if (j.contains("radius_m")) number_in(j, path, "radius_m", kMinRadiusM, kMaxRadiusM);
    number_in(j, path, "altitude_m", kMinAltM, kMaxAltM);
    if (present(j, "duration_s")) integer_in(j, path, "duration_s", 10, 6 * 3600);
  } else if (a == "set_speed") {
    check_keys(j, path, {"action", "speed_ms"}, {});
    number_in(j, path, "speed_ms", kMinSpeedMs, kMaxSpeedMs);
  } else if (a == "return_to_launch") {
    check_keys(j, path, {"action"}, {});
  } else if (a == "land") {
    check_keys(j, path, {"action"}, {"at"});
    if (present(j, "at")) location(j["at"], join(path, "at"));
  } else {
    fail(join(path, "action"), "bilinmeyen aksiyon: " + a);
  }
}

std::string validate_impl(const json& doc) {
  require_object(doc, "");
  if (!doc.contains("type") || !doc["type"].is_string()) fail("type", "ayırıcı eksik");
  const auto type = doc["type"].get<std::string>();
  if (type == "plan") {
    check_keys(doc, "", {"type", "steps"}, {});
    const auto& s = doc["steps"];
    if (!s.is_array() || s.empty() || s.size() > static_cast<size_t>(kMaxSteps))
      fail("steps", "1-20 öğeli liste bekleniyordu");
    for (size_t i = 0; i < s.size(); ++i) action(s[i], "steps[" + std::to_string(i) + "]");
  } else if (type == "clarification") {
    check_keys(doc, "", {"type", "question", "missing"}, {});
    string_len(doc["question"], "question", 3, 300);
    const auto& m = doc["missing"];
    if (!m.is_array() || m.empty() || m.size() > 5) fail("missing", "1-5 öğeli liste bekleniyordu");
    for (size_t i = 0; i < m.size(); ++i) {
      const std::string p = "missing[" + std::to_string(i) + "]";
      if (!m[i].is_string()) fail(p, "metin bekleniyordu");
      if (blank(m[i].get<std::string>())) fail(p, "boş olamaz");
    }
  } else if (type == "refusal") {
    check_keys(doc, "", {"type", "reason"}, {});
    string_len(doc["reason"], "reason", 3, 300);
  } else {
    fail("type", "bilinmeyen tür: " + type);
  }
  return type;
}

}  // namespace

Validation validate_output(const nlohmann::json& doc) {
  Validation v;
  try {
    v.type = validate_impl(doc);
    v.ok = true;
  } catch (const SchemaError& e) {
    v.error = e.what();
  }
  return v;
}

Validation validate_text(const std::string& text) {
  nlohmann::json doc = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded()) {
    Validation v;
    v.error = "geçerli JSON değil";
    return v;
  }
  return validate_output(doc);
}

}  // namespace skymind
