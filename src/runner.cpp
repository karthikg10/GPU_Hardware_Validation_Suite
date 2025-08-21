// src/runner.cpp
#include <thread>
#include <mutex>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "runner_iface.h"

extern "C" void memory_test_entry(int, size_t, int, double&, double&, double&, int&);
extern "C" void compute_test_entry(int, size_t, int, double&, int&);
extern "C" void interconnect_test_entry(size_t, double&, bool&);

static std::mutex print_mtx;

static void run_on_device(int dev, const RunConfig& cfg, DeviceResult& out){
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  {
    std::lock_guard<std::mutex> lk(print_mtx);
    std::printf("[GPU %d] %s, %zu MiB global mem\n", dev, p.name, p.totalGlobalMem>>20);
  }
  if(cfg.do_mem){
    memory_test_entry(dev, cfg.mem_bytes, cfg.streams, out.h2d, out.d2h, out.devbw, out.mem_errors);
  }
  if(cfg.do_compute){
    compute_test_entry(dev, cfg.compute_elems, cfg.compute_iters, out.gflops, out.comp_errors);
  }
  out.device = dev;
}

std::vector<DeviceResult> run_all(const RunConfig& cfg){
  int n=0; CUDA_CHECK(cudaGetDeviceCount(&n));
  std::vector<DeviceResult> results(n);
  std::vector<std::thread> threads;
  for(int d=0; d<n; ++d){
    threads.emplace_back([&, d]{ run_on_device(d, cfg, results[d]); });
  }
  for(auto& t: threads) t.join();

  if(cfg.do_interconnect){
    double p2p=0; bool used=false; interconnect_test_entry(cfg.interconnect_bytes, p2p, used);
    std::lock_guard<std::mutex> lk(print_mtx);
    if(used) std::printf("{\"interconnect\":{\"p2p_GBs\":%.2f}}\n", p2p);
    else     std::printf("{\"interconnect\":{\"p2p_GBs\":null, \"note\":\"single GPU or no P2P\"}}\n");
  }

  for(const auto& r: results){
    std::printf("{\"gpu\":%d,\"mem\":{\"h2d_GBs\":%.2f,\"d2h_GBs\":%.2f,\"dev_GBs\":%.2f,\"errors\":%d},"
                "\"compute\":{\"GFLOPs\":%.2f,\"errors\":%d}}\n",
                r.device, r.h2d, r.d2h, r.devbw, r.mem_errors, r.gflops, r.comp_errors);
  }
  return results;
}
