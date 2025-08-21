#include <cstdio>
#include <vector>
#include "cuda_utils.h"
#include "timers.h"
#include "nvtx_scopes.h"

struct InterconnectResult { double p2pGBs{}; bool usedP2P{}; };

InterconnectResult run_interconnect_test(size_t bytes){
  NvtxRange r("interconnect_test");
  int nDevs=0; CUDA_CHECK(cudaGetDeviceCount(&nDevs));
  if(nDevs < 2) return {0.0, false};

  CUDA_CHECK(cudaSetDevice(0));
  void* d0=nullptr; CUDA_CHECK(cudaMalloc(&d0, bytes));
  CUDA_CHECK(cudaSetDevice(1));
  void* d1=nullptr; CUDA_CHECK(cudaMalloc(&d1, bytes));

  int can01=0, can10=0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
  if(can01) { CUDA_CHECK(cudaSetDevice(0)); cudaDeviceEnablePeerAccess(1,0); }
  if(can10) { CUDA_CHECK(cudaSetDevice(1)); cudaDeviceEnablePeerAccess(0,0); }
  if(!(can01 && can10)) {
    CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaFree(d0));
    CUDA_CHECK(cudaSetDevice(1)); CUDA_CHECK(cudaFree(d1));
    return {0.0, false};
  }

  cudaStream_t s0; CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaStreamCreate(&s0));
  GpuTimer t; t.tic(s0);
  CUDA_CHECK(cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s0));
  CUDA_CHECK(cudaStreamSynchronize(s0));
  float ms = t.toc(s0);

  CUDA_CHECK(cudaStreamDestroy(s0));
  CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaFree(d0));
  CUDA_CHECK(cudaSetDevice(1)); CUDA_CHECK(cudaFree(d1));

  double gbps = (bytes/1e9)/(ms/1e3);
  return {gbps, true};
}

extern "C" void interconnect_test_entry(size_t bytes, double& p2pGBs, bool& usedP2P){
  auto r = run_interconnect_test(bytes);
  p2pGBs = r.p2pGBs; usedP2P = r.usedP2P;
}
