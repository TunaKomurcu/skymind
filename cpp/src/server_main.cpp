// SkyMind C++ çıkarım servisi.
//
//   skymind_server -m model.gguf [--port 8080] [--ngl 0] [--threads N] [--system configs/system_short.txt]
//
// Uçlar:  POST /v1/plan {"command": "..."}   GET /healthz   GET /metrics (Prometheus)
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include <httplib.h>

#include "skymind/llama_engine.hpp"
#include "skymind/metrics.hpp"
#include "skymind/prompt.hpp"
#include "skymind/service.hpp"

namespace {

struct Args {
  skymind::LlamaOptions llama;
  std::string system_path = "cpp/configs/system_short.txt";
  std::string host = "127.0.0.1";
  int port = 8080;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
               "kullanim: %s -m model.gguf [--host 127.0.0.1] [--port 8080] [--ngl N] [--threads N]\n"
               "          [--ctx 2048] [--max-new-tokens 400] [--system dosya] [--verbose]\n",
               argv0);
}

bool parse(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    std::string v;
    if (k == "-m" || k == "--model") {
      if (!next(a.llama.model_path)) return false;
    } else if (k == "--host") {
      if (!next(a.host)) return false;
    } else if (k == "--port") {
      if (!next(v)) return false;
      a.port = std::atoi(v.c_str());
    } else if (k == "--ngl") {
      if (!next(v)) return false;
      a.llama.n_gpu_layers = std::atoi(v.c_str());
    } else if (k == "--threads") {
      if (!next(v)) return false;
      a.llama.n_threads = std::atoi(v.c_str());
    } else if (k == "--ctx") {
      if (!next(v)) return false;
      a.llama.n_ctx = std::atoi(v.c_str());
    } else if (k == "--max-new-tokens") {
      if (!next(v)) return false;
      a.llama.max_new_tokens = std::atoi(v.c_str());
    } else if (k == "--system") {
      if (!next(a.system_path)) return false;
    } else if (k == "--verbose") {
      a.llama.quiet = false;
    } else {
      return false;
    }
  }
  return !a.llama.model_path.empty() && a.port > 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse(argc, argv, args)) {
    usage(argv[0]);
    return 2;
  }
  try {
    const std::string system = skymind::read_system_prompt(args.system_path);
    auto engine = std::make_shared<skymind::LlamaEngine>(args.llama);
    skymind::Metrics metrics;
    metrics.describe("skymind_model_info", "gauge", "Yuklu model (etiket)");
    metrics.set_gauge("skymind_model_info", 1, "engine=\"" + engine->name() + "\"");
    skymind::PlannerService service(engine, system, metrics);

    httplib::Server svr;
    svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
      res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
      res.set_content(metrics.render(), "text/plain; version=0.0.4");
    });
    svr.Post("/v1/plan", [&](const httplib::Request& req, httplib::Response& res) {
      const skymind::PlanResponse r = service.plan(req.body);
      res.status = r.http_status;
      res.set_content(r.body.dump(), "application/json; charset=utf-8");
    });

    std::fprintf(stderr, "skymind_server hazir: http://%s:%d  (%s)\n", args.host.c_str(), args.port,
                 engine->name().c_str());
    if (!svr.listen(args.host, args.port)) {
      std::fprintf(stderr, "port acilamadi: %d\n", args.port);
      return 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hata: %s\n", e.what());
    return 1;
  }
  return 0;
}
