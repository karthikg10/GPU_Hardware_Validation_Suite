#include <cstdio>
#include <cmath>
#include "cuda_utils.h"
#include "timers.h"
#include "nvtx_scopes.h"

__global__ void fma_stress(float* out, int iters){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = (float)i, y = 1.0001f, z = 0.9997f;
  #pragma unroll 4
  for(int k=0;k<iters;k++) x = fmaf(x, y, z);
  out[i] = x;
}

struct ComputeResult { double gflops{}; int errors{}; };

ComputeResult run_compute_test(int device, size_t elements, int iters){
  NvtxRange r("compute_test");
  CUDA_CHECK(cudaSetDevice(device));
  float* d=nullptr; CUDA_CHECK(cudaMalloc(&d, elements*sizeof(float)));
  dim3 block(256), grid((unsigned)((elements+block.x-1)/block.x));

  GpuTimer t; t.tic();
  fma_stress<<<grid, block>>>(d, iters);
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.toc();

  double flops = double(elements) * double(iters) * 2.0; // 1 FMA = 2 flops
  double gflops = (flops/1e9) / (ms/1e3);

  float h[4]; CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  int errors = 0;
  for (float v : h) if (!std::isfinite(v)) errors++;

  CUDA_CHECK(cudaFree(d));
  return {gflops, errors};
}

extern "C" void compute_test_entry(int device, size_t elements, int iters,
                                   double& gflops, int& errors){
  auto r = run_compute_test(device, elements, iters);
  gflops = r.gflops; errors = r.errors;
}
