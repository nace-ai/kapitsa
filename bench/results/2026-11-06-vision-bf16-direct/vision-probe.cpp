// Probe the actual VitWeights reader without constructing a GPU. close() releases the headers;
// _Exit skips VitWeights::release(), whose destructor initializes a Metal device even with no owned buffers.
#include "vision/vit_common.h"
#include "util/bf16.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <unistd.h>
using namespace ni;
int main(int argc,char**argv) {
  if(argc!=6) return 2;
  std::string dir=argv[1],name=argv[2];size_t rows=std::stoull(argv[3]),k=std::stoull(argv[4]),ld=std::stoull(argv[5]);
  auto* w=new vision::VitWeights;w->open(dir,"");
  std::vector<uint16_t> out(rows*ld+2,0xa55a);
  auto start=std::chrono::steady_clock::now();w->read_bf16(name,out.data()+1,rows,k,ld);
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  rusage ru{};getrusage(RUSAGE_SELF,&ru);
  if(out.front()!=0xa55a || out.back()!=0xa55a) return 3;
  for(size_t r=0;r<rows;++r) for(size_t c=k;c<ld;++c) if(out[1+r*ld+c]) return 4;
  uint64_t hash=14695981039346656037ull;
  for(size_t i=1;i+1<out.size();++i) { hash^=out[i];hash*=1099511628211ull; }
  // Optional raw output supplies a complete byte comparison; hashes alone are not the output gate.
  if(const char* file=getenv("PROBE_OUT")) {std::ofstream f(file,std::ios::binary);f.write((char*)(out.data()+1),rows*ld*2);if(!f)return 5;}
  w->close();
  nlohmann::json j{{"rows",rows},{"k",k},{"ld",ld},{"ms",ms},{"peak_rss_bytes",ru.ru_maxrss},{"hash",hash},{"guards",true},{"padding_zero",true}};
  std::cout<<j.dump()<<std::endl;std::_Exit(0);
}
