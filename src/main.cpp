#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include "runner_iface.h"

int main(int argc, char** argv){
  RunConfig cfg;
  for(int i=1;i<argc;i++){
    if(!std::strcmp(argv[i],"--mem-bytes") && i+1<argc) cfg.mem_bytes = std::stoull(argv[++i]);
    else if(!std::strcmp(argv[i],"--streams") && i+1<argc) cfg.streams = std::stoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--compute-elems") && i+1<argc) cfg.compute_elems = std::stoull(argv[++i]);
    else if(!std::strcmp(argv[i],"--compute-iters") && i+1<argc) cfg.compute_iters = std::stoi(argv[++i]);
    else if(!std::strcmp(argv[i],"--no-mem"))          cfg.do_mem=false;
    else if(!std::strcmp(argv[i],"--no-compute"))      cfg.do_compute=false;
    else if(!std::strcmp(argv[i],"--no-interconnect")) cfg.do_interconnect=false;
    else if(!std::strcmp(argv[i],"--help")){
      std::cout <<
      "gpuval options:\n"
      "  --mem-bytes <N>        memory test size (bytes)\n"
      "  --streams <N>          memory test streams (default 4)\n"
      "  --compute-elems <N>    compute elements\n"
      "  --compute-iters <N>    compute iterations\n"
      "  --no-mem | --no-compute | --no-interconnect\n";
      return 0;
    }
  }
  run_all(cfg);
  return 0;
}
