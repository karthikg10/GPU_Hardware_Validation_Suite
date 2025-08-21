// src/cupti_stub.h
#pragma once
struct TraceConfig { bool enable=false; };
void trace_init(const TraceConfig&){ /* no-op stub */ }
void trace_shutdown(){ /* no-op stub */ }
