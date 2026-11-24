#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "skymind/service.hpp"

using namespace skymind;

namespace {

class FakeEngine : public InferenceEngine {
 public:
  explicit FakeEngine(std::string reply, bool throws = false)
      : reply_(std::move(reply)), throws_(throws) {}
  GenerationResult generate(const std::string& system, const std::string& command) override {
    last_system = system;
    last_command = command;
    if (throws_) throw std::runtime_error("model coktu");
    GenerationResult g;
    g.text = reply_;
    g.prompt_tokens = 10;
    g.new_tokens = 5;
    return g;
  }
  std::string name() const override { return "fake"; }
  std::string last_system, last_command;

 private:
  std::string reply_;
  bool throws_;
};

const char* kPlan = R"({"type":"plan","steps":[{"action":"takeoff","altitude_m":100}]})";

}  // namespace

TEST(Service, ValidPlanIsReturnedParsed) {
  Metrics m;
  auto eng = std::make_shared<FakeEngine>(kPlan);
  PlannerService svc(eng, "SYS", m);
  const auto r = svc.plan(R"({"command":"100 metreye kalk"})");
  EXPECT_EQ(r.http_status, 200);
  EXPECT_TRUE(r.body["valid"].get<bool>());
  EXPECT_EQ(r.body["type"], "plan");
  EXPECT_EQ(r.body["plan"]["steps"][0]["action"], "takeoff");
  EXPECT_EQ(eng->last_system, "SYS");
  EXPECT_EQ(eng->last_command, "100 metreye kalk");
  const auto text = m.render();
  EXPECT_NE(text.find("skymind_requests_total{status=\"ok\"} 1"), std::string::npos) << text;
  EXPECT_NE(text.find("skymind_output_total{type=\"plan\"} 1"), std::string::npos) << text;
  EXPECT_NE(text.find("skymind_generated_tokens_total 5"), std::string::npos) << text;
}

TEST(Service, InvalidModelOutputIsFlaggedNotHidden) {
  Metrics m;
  PlannerService svc(std::make_shared<FakeEngine>(R"({"type":"plan","steps":[]})"), "S", m);
  const auto r = svc.plan(R"({"command":"x"})");
  EXPECT_EQ(r.http_status, 200);
  EXPECT_FALSE(r.body["valid"].get<bool>());
  EXPECT_TRUE(r.body["plan"].is_null());
  EXPECT_FALSE(r.body["validation_error"].is_null());
  EXPECT_EQ(r.body["raw"], R"({"type":"plan","steps":[]})");
  EXPECT_NE(m.render().find("skymind_output_total{type=\"invalid\"} 1"), std::string::npos);
}

TEST(Service, BadRequestsGet400AndAreCounted) {
  Metrics m;
  PlannerService svc(std::make_shared<FakeEngine>(kPlan), "S", m);
  EXPECT_EQ(svc.plan("not json").http_status, 400);
  EXPECT_EQ(svc.plan(R"({"cmd":"x"})").http_status, 400);
  EXPECT_EQ(svc.plan(R"({"command":""})").http_status, 400);
  EXPECT_EQ(svc.plan(R"({"command":42})").http_status, 400);
  EXPECT_EQ(svc.plan("{\"command\":\"" + std::string(2001, 'a') + "\"}").http_status, 400);
  EXPECT_NE(m.render().find("status=\"bad_request\"} 5"), std::string::npos) << m.render();
}

TEST(Service, EngineFailureGives500AndInFlightReturnsToZero) {
  Metrics m;
  PlannerService svc(std::make_shared<FakeEngine>("", /*throws=*/true), "S", m);
  const auto r = svc.plan(R"({"command":"x"})");
  EXPECT_EQ(r.http_status, 500);
  EXPECT_NE(r.body["error"].get<std::string>().find("model coktu"), std::string::npos);
  const auto text = m.render();
  EXPECT_NE(text.find("skymind_requests_in_flight 0"), std::string::npos) << text;
  EXPECT_NE(text.find("status=\"error\"} 1"), std::string::npos) << text;
}
