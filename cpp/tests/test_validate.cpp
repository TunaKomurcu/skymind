#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "skymind/validate.hpp"

using nlohmann::json;
using skymind::validate_output;
using skymind::validate_text;

// Python (Pydantic) ile aynı karar: cases.jsonl'deki "valid" alanı Pydantic'ten gelir.
TEST(Parity, AgreesWithPydanticOnEveryCase) {
  std::ifstream f(SKYMIND_TEST_DATA "/cases.jsonl", std::ios::binary);
  ASSERT_TRUE(f.good()) << "fikstür yok: python scripts/export_cpp_fixtures.py";
  std::string line;
  int n = 0, valid = 0, disagreements = 0;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const json c = json::parse(line);
    const auto v = validate_output(c["doc"]);
    const bool expected = c["valid"].get<bool>();
    ++n;
    valid += expected;
    if (v.ok != expected) {
      ++disagreements;
      ADD_FAILURE() << "ayrışma (beklenen " << expected << ", C++ " << v.ok << "): "
                    << c["doc"].dump() << "\n  C++ hatası: " << v.error;
    }
  }
  EXPECT_GT(n, 200);
  EXPECT_GT(valid, 50);
  EXPECT_GT(n - valid, 50);  // hem geçerli hem geçersiz örnek var
  EXPECT_EQ(disagreements, 0);
}

TEST(Validate, ReportsPathOfFirstError) {
  const auto v = validate_text(
      R"({"type":"plan","steps":[{"action":"takeoff","altitude_m":100},{"action":"takeoff","altitude_m":6000}]})");
  EXPECT_FALSE(v.ok);
  EXPECT_NE(v.error.find("steps[1].altitude_m"), std::string::npos) << v.error;
}

TEST(Validate, ReturnsTypeWhenValid) {
  EXPECT_EQ(validate_text(R"({"type":"refusal","reason":"kapsam dışı"})").type, "refusal");
  EXPECT_EQ(validate_text(R"({"type":"clarification","question":"Hangi irtifa?","missing":["alt"]})").type,
            "clarification");
}

TEST(Validate, NotJsonIsInvalid) {
  EXPECT_FALSE(validate_text("merhaba").ok);
  EXPECT_FALSE(validate_text("").ok);
  EXPECT_FALSE(validate_text("[1,2]").ok);
}

TEST(Validate, BooleanIsNotANumber) {
  EXPECT_FALSE(validate_text(R"({"type":"plan","steps":[{"action":"takeoff","altitude_m":true}]})").ok);
}
