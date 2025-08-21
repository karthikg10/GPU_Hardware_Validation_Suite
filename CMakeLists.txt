# CMakeLists.txt — rock-solid for CMake 3.16
cmake_minimum_required(VERSION 3.16)
project(gpuval LANGUAGES CXX CUDA)

# ---- Hard reset any inherited CUDA standard knobs ----
unset(CMAKE_CUDA_STANDARD CACHE)
unset(CMAKE_CUDA_STANDARD_REQUIRED CACHE)
unset(CMAKE_CUDA_EXTENSIONS CACHE)
set(CMAKE_CUDA_STANDARD 14 CACHE STRING "CUDA device standard" FORCE)

# ---- Host side ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")

# ---- GPU arch handling ----
# Default to T4 (sm_75); override via: -DGPU_ARCHS="75" (or "86;89;90", etc.)
if(NOT DEFINED GPU_ARCHS)
  set(GPU_ARCHS "75")
endif()

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(GENCODE_FLAGS "")
  foreach(arch IN LISTS GPU_ARCHS)
    list(APPEND GENCODE_FLAGS "-gencode=arch=compute_${arch},code=sm_${arch}")
  endforeach()
endif()

# ---- Sources ----
set(SRC
  src/main.cpp
  src/runner.cpp
  src/tests_memory.cu
  src/tests_compute.cu
  src/tests_interconnect.cu
  src/cupti_stub.cpp
)

add_executable(gpuval ${SRC})
target_include_directories(gpuval PRIVATE include)
target_compile_features(gpuval PRIVATE cxx_std_17)

# Ensure NVCC uses a valid device standard on old CMake
target_compile_options(gpuval PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++14>
)

# Apply arch flags (old/new paths)
if(CMAKE_VERSION VERSION_LESS 3.18)
  target_compile_options(gpuval PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${GENCODE_FLAGS}>
  )
else()
  set_target_properties(gpuval PROPERTIES
    CUDA_ARCHITECTURES "${GPU_ARCHS}"
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

# ---- Optional NVTX (won't fail if missing) ----
find_path(NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
  PATHS "$ENV{CUDA_PATH}/include" "/usr/local/cuda/include")
find_library(NVTX_LIBRARY nvToolsExt
  PATHS "$ENV{CUDA_PATH}/lib64" "/usr/local/cuda/lib64")
if(NVTX_INCLUDE_DIR AND NVTX_LIBRARY)
  target_include_directories(gpuval PRIVATE ${NVTX_INCLUDE_DIR})
  target_link_libraries(gpuval PRIVATE ${NVTX_LIBRARY})
  target_compile_definitions(gpuval PRIVATE USE_NVTX=1)
endif()
