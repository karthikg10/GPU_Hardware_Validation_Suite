// include/nvtx_scopes.h
#pragma once
#ifdef USE_NVTX
  #include <nvtx3/nvToolsExt.h>
  struct NvtxRange {
    nvtxRangeId_t id;
    NvtxRange(const char* msg) { id = nvtxRangeStartA(msg); }
    ~NvtxRange() { nvtxRangeEnd(id); }
  };
#else
  struct NvtxRange { NvtxRange(const char*) {} };
#endif
