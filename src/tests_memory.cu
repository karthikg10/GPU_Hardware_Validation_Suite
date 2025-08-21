#include <vector>
#include <cstdio>
#include <cstdint>
#include "cuda_utils.h"
#include "timers.h"
#include "nvtx_scopes.h"

__global__ void fill_pattern(uint32_t* p, size_t n, uint32_t seed){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n) p[i] = (uint32_t)(i*1664525u + seed);
}
__global__ void verify_pattern(const uint32_t* p, size_t n, uint32_t seed, int* errors){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n){
    uint32_t expect = (uint32_t)(i*1664525u + seed);
    if(p[i] != expect) atomicAdd(errors, 1);
  }
}

struct MemoryResult { double h2dGBs{}, d2hGBs{}, deviceGBs{}; int errors{}; };

MemoryResult run_memory_test(int device, size_t bytes, int streams){
  NvtxRange r("memory_test");
  CUDA_CHECK(cudaSetDevice(device));

  void* hbuf=nullptr;  CUDA_CHECK(cudaMallocHost(&hbuf,  bytes));
  void* hbuf2=nullptr; CUDA_CHECK(cudaMallocHost(&hbuf2, bytes));
  void* dbuf=nullptr;  CUDA_CHECK(cudaMalloc(&dbuf, bytes));

  std::vector<cudaStream_t> ss(streams);
  for(auto& s : ss) CUDA_CHECK(cudaStreamCreate(&s));

  size_t chunk = bytes/streams;

  // H2D bandwidth
  GpuTimer t; t.tic();
  for(int i=0;i<streams;i++){
    char* h = (char*)hbuf + i*chunk;
    char* d = (char*)dbuf + i*chunk;
    CUDA_CHECK(cudaMemcpyAsync(d, h, chunk, cudaMemcpyHostToDevice, ss[i]));
  }
  for(auto s: ss) CUDA_CHECK(cudaStreamSynchronize(s));
  float ms_h2d = t.toc();
  double h2dGBs = (bytes/1e9)/(ms_h2d/1e3);

  // Device fill + verify
  int* d_err=nullptr; CUDA_CHECK(cudaMalloc(&d_err, sizeof(int)));
  CUDA_CHECK(cudaMemsetAsync(d_err, 0, sizeof(int), ss[0]));
  size_t n = bytes/sizeof(uint32_t);
  dim3 block(256), grid((unsigned)((n+block.x-1)/block.x));
  t.tic(ss[0]);
  fill_pattern<<<grid, block, 0, ss[0]>>>((uint32_t*)dbuf, n, 123u);
  verify_pattern<<<grid, block, 0, ss[0]>>>((uint32_t*)dbuf, n, 123u, d_err);
  CUDA_CHECK(cudaStreamSynchronize(ss[0]));
  float ms_dev = t.toc(ss[0]);
  int h_err=0; CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_err));

  // D2H bandwidth
  t.tic();
  for(int i=0;i<streams;i++){
    char* h = (char*)hbuf2 + i*chunk;
    char* d = (char*)dbuf + i*chunk;
    CUDA_CHECK(cudaMemcpyAsync(h, d, chunk, cudaMemcpyDeviceToHost, ss[i]));
  }
  for(auto s: ss) CUDA_CHECK(cudaStreamSynchronize(s));
  float ms_d2h = t.toc();
  double d2hGBs = (bytes/1e9)/(ms_d2h/1e3);

  for(auto s: ss) cudaStreamDestroy(s);
  CUDA_CHECK(cudaFree(dbuf));
  CUDA_CHECK(cudaFreeHost(hbuf));
  CUDA_CHECK(cudaFreeHost(hbuf2));

  return {h2dGBs, d2hGBs, (bytes/1e9)/(ms_dev/1e3), h_err};
}

extern "C" void memory_test_entry(int device, size_t bytes, int streams,
                                  double& h2d, double& d2h, double& dgbps, int& errors) {
  auto r = run_memory_test(device, bytes, streams);
  h2d = r.h2dGBs; d2h = r.d2hGBs; dgbps = r.deviceGBs; errors = r.errors;
}
