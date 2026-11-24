#include "skymind/metrics.hpp"

#include <cstdio>
#include <sstream>

namespace skymind {
namespace {

std::string fmt(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  return buf;
}

std::string with_labels(const std::string& name, const std::string& labels) {
  return labels.empty() ? name : name + "{" + labels + "}";
}

}  // namespace

Metrics::Metrics() : bounds_{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10} {}

void Metrics::inc(const std::string& name, const std::string& labels, double by) {
  std::lock_guard<std::mutex> lock(mu_);
  series_[name][labels] += by;
}

void Metrics::set_gauge(const std::string& name, double value, const std::string& labels) {
  std::lock_guard<std::mutex> lock(mu_);
  series_[name][labels] = value;
}

void Metrics::add_gauge(const std::string& name, double delta, const std::string& labels) {
  std::lock_guard<std::mutex> lock(mu_);
  series_[name][labels] += delta;
}

void Metrics::describe(const std::string& name, const std::string& type, const std::string& help) {
  std::lock_guard<std::mutex> lock(mu_);
  meta_[name] = {type, help};
}

void Metrics::observe(const std::string& name, double value) {
  std::lock_guard<std::mutex> lock(mu_);
  Hist& h = hists_[name];
  if (h.counts.empty()) h.counts.assign(bounds_.size() + 1, 0.0);
  size_t i = 0;
  while (i < bounds_.size() && value > bounds_[i]) ++i;  // le: değer <= sınır
  h.counts[i] += 1;
  h.sum += value;
  h.total += 1;
}

std::string Metrics::render() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::ostringstream out;
  auto head = [&](const std::string& name) {
    auto it = meta_.find(name);
    if (it == meta_.end()) return;
    out << "# HELP " << name << " " << it->second.second << "\n";
    out << "# TYPE " << name << " " << it->second.first << "\n";
  };
  for (const auto& [name, by_label] : series_) {
    head(name);
    for (const auto& [labels, value] : by_label)
      out << with_labels(name, labels) << " " << fmt(value) << "\n";
  }
  for (const auto& [name, h] : hists_) {
    head(name);
    double cum = 0;
    for (size_t i = 0; i < bounds_.size(); ++i) {
      cum += h.counts[i];
      out << name << "_bucket{le=\"" << fmt(bounds_[i]) << "\"} " << fmt(cum) << "\n";
    }
    out << name << "_bucket{le=\"+Inf\"} " << h.total << "\n";
    out << name << "_sum " << fmt(h.sum) << "\n";
    out << name << "_count " << h.total << "\n";
  }
  return out.str();
}

}  // namespace skymind
