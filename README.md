# GPU Hardware Validation Suite (C++/CUDA)
Checks GPU memory bandwidth, compute throughput (FMA), and P2P interconnect (if available). Emits JSON for baselining.

## Build
cmake -S . -B build -DGPU_ARCHS=75
cmake --build build -j

## Run
./build/gpuval --bytes 512M --streams 8

## Example output
{"interconnect":{"p2p_GBs":null,"note":"single GPU or no P2P"}}
{"gpu":0,"mem":{"h2d_GBs":6.26,"d2h_GBs":6.60,"dev_GBs":114.34,"errors":0},"compute":{"GFLOPs":7309.29,"errors":0}}
## Notes
- Works on **single GPU**; interconnect is reported as “not applicable” if no P2P.
- Time is measured with **CUDA events** on the stream under test.
- Use `-DGPU_ARCHS=<sm>` (e.g., 75=T4, 80=A100, 89=H100) at configure time.
