#include <string>

#include <gtest/gtest.h>

#include "skymind/metrics.hpp"
#include "skymind/prompt.hpp"

using skymind::Metrics;

TEST(Prompt, ChatMlMatchesQwenTemplate) {
  EXPECT_EQ(skymind::chatml_prompt("SYS", "kalk"),
            "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nkalk<|im_end|>\n<|im_start|>assistant\n");
}

TEST(Prompt, SystemPromptFileIsReadWithoutTrailingNewline) {
  const std::string s = skymind::read_system_prompt(SKYMIND_CONFIGS "/system_short.txt");
  EXPECT_FALSE(s.empty());
  EXPECT_NE(s.back(), '\n');
  EXPECT_NE(s.find("JSON"), std::string::npos);
}

TEST(Prompt, MissingFileThrows) {
  EXPECT_THROW(skymind::read_system_prompt("yok/yok.txt"), std::runtime_error);
}

TEST(Metrics, CountersAccumulatePerLabelSet) {
  Metrics m;
  m.describe("req_total", "counter", "istekler");
  m.inc("req_total", "status=\"ok\"");
  m.inc("req_total", "status=\"ok\"");
  m.inc("req_total", "status=\"error\"", 3);
  const std::string out = m.render();
  EXPECT_NE(out.find("# TYPE req_total counter"), std::string::npos);
  EXPECT_NE(out.find("req_total{status=\"ok\"} 2\n"), std::string::npos) << out;
  EXPECT_NE(out.find("req_total{status=\"error\"} 3\n"), std::string::npos) << out;
}

TEST(Metrics, GaugeSetAndAdd) {
  Metrics m;
  m.set_gauge("inflight", 0);
  m.add_gauge("inflight", 2);
  m.add_gauge("inflight", -1);
  EXPECT_NE(m.render().find("inflight 1\n"), std::string::npos);
}

TEST(Metrics, HistogramBucketsAreCumulative) {
  Metrics m;
  m.observe("lat", 0.04);  // <=0.05
  m.observe("lat", 0.3);   // <=0.5
  m.observe("lat", 20.0);  // sadece +Inf
  const std::string out = m.render();
  EXPECT_NE(out.find("lat_bucket{le=\"0.05\"} 1\n"), std::string::npos) << out;
  EXPECT_NE(out.find("lat_bucket{le=\"0.5\"} 2\n"), std::string::npos) << out;
  EXPECT_NE(out.find("lat_bucket{le=\"10\"} 2\n"), std::string::npos) << out;
  EXPECT_NE(out.find("lat_bucket{le=\"+Inf\"} 3\n"), std::string::npos) << out;
  EXPECT_NE(out.find("lat_count 3\n"), std::string::npos) << out;
  EXPECT_NE(out.find("lat_sum 20.34\n"), std::string::npos) << out;
}
