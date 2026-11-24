// Bağımlılıksız Prometheus metrik kaydı (counter, gauge, histogram) ve metin biçimi çıktısı.
#pragma once

#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace skymind {

class Metrics {
 public:
  Metrics();

  // Sayaç: ad + etiket kümesi başına artar. labels örn. status="ok" (boş olabilir).
  void inc(const std::string& name, const std::string& labels = "", double by = 1.0);
  void set_gauge(const std::string& name, double value, const std::string& labels = "");
  void add_gauge(const std::string& name, double delta, const std::string& labels = "");
  void observe(const std::string& name, double value);  // gecikme histogramı (saniye)

  void describe(const std::string& name, const std::string& type, const std::string& help);

  // Prometheus text exposition format 0.0.4
  std::string render() const;

 private:
  struct Hist {
    std::vector<double> counts;  // kümülatif olmayan kova sayıları; son eleman +Inf
    double sum = 0;
    unsigned long total = 0;
  };
  mutable std::mutex mu_;
  std::map<std::string, std::map<std::string, double>> series_;      // ad -> etiket -> değer
  std::map<std::string, std::pair<std::string, std::string>> meta_;  // ad -> (tür, yardım)
  std::map<std::string, Hist> hists_;
  std::vector<double> bounds_;
};

}  // namespace skymind
