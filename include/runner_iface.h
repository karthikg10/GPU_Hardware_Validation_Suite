#pragma once
#include <cstddef>
#include <vector>

// Config the CLI fills in
struct RunConfig {
  size_t mem_bytes = size_t(1) << 28;   // 256 MiB
  int    streams   = 4;
  size_t compute_elems = 1 << 22;       // ~4M
  int    compute_iters = 2000;
  size_t interconnect_bytes = size_t(1) << 28;
  bool   do_mem = true, do_compute = true, do_interconnect = true;
};

// Per-GPU result (runner emits JSON lines; you don't need this in main,
// but it's handy to keep the API complete)
struct DeviceResult {
  int device{};
  double h2d{}, d2h{}, devbw{}; int mem_errors{};
  double gflops{}; int comp_errors{};
};

// Launch all tests across visible GPUs
std::vector<DeviceResult> run_all(const RunConfig& cfg);
