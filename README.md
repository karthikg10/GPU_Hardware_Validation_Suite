# GPU Hardware Validation Suite (C++/CUDA)
Checks GPU memory bandwidth, compute throughput (FMA), and P2P interconnect (if available). Emits JSON for baselining.

## Build
cmake -S . -B build -DGPU_ARCHS=75
cmake --build build -j

## Run
./build/gpuval --bytes 512M --streams 8
