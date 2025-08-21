// include/cuda_utils.h
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
inline void checkCuda(cudaError_t e, const char* call, const char* file, int line) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e) +
      " at " + file + ":" + std::to_string(line) + " in call " + call);
  }
}
#define CUDA_CHECK(x) checkCuda((x), #x, __FILE__, __LINE__)
