#include "gguf/quant.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>

using Pack = void(*)(const uint8_t*, size_t, int, uint32_t*);
void reference_pack(const uint8_t*, size_t, int, uint32_t*);
static void require(bool ok) { if (!ok) { puts("FAIL"); std::exit(1); } }
int main(int argc,char**argv) {
  require(argc==2);
  constexpr size_t n=2560, rows=1024;
  std::ifstream file(argv[1],std::ios::binary);
  require(file.good());
  const std::vector<uint8_t> raw((std::istreambuf_iterator<char>(file)),{});
  require(raw.size()==rows*n/32*18);
  std::vector<float> x(rows*n);
  ni::gguf::dequantize(20,raw.data(),x.data(),x.size());
  for(float f:x)require(std::isfinite(f));
  {
    std::vector<uint8_t> codes(n*rows);
    std::vector<uint16_t> sc(n*rows/64),bi(sc.size());
    std::vector<uint32_t> a(n*rows*5/32),b(a.size());
    ni::gguf::quantize(x.data(),x.size(),5,64,codes.data(),sc.data(),bi.data());
    reference_pack(codes.data(),codes.size(),5,a.data());
    ni::gguf::pack_codes(codes.data(),codes.size(),5,b.data());require(a==b);
    const auto stableCodes=codes;const auto stableSc=sc;const auto stableBi=bi;
    for(bool full: {false,true}) for(int leg=0;leg<8;++leg) {
      Pack pack=leg%2?ni::gguf::pack_codes:reference_pack;
      const int passes=full?4:16;
      const auto begin=std::chrono::steady_clock::now();
      for(int pass=0;pass<passes;++pass)for(size_t row=0;row<rows;++row) {
        if(full)ni::gguf::quantize(x.data()+row*n,n,5,64,codes.data()+row*n,sc.data()+row*n/64,bi.data()+row*n/64);
        pack(codes.data()+row*n,n,5,b.data()+row*n*5/32);
      }
      const auto end=std::chrono::steady_clock::now();
      require(a==b && codes==stableCodes && sc==stableSc && bi==stableBi);
      const double ns=std::chrono::duration<double,std::nano>(end-begin).count()/(rows*passes);
      printf("{\"width\":%zu,\"mode\":\"%s\",\"leg\":%d,\"arm\":\"%s\",\"ns_per_row\":%.3f,\"rows\":%zu,\"passes\":%d,\"full_bytes_gate\":true}\n",n,full?"quantize_and_pack":"pack",leg+1,leg%2?"B":"A",ns,rows,passes);
      fflush(stdout);
    }
  }
}
