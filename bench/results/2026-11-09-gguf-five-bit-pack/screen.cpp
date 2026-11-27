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

using Pack = void(*)(const uint8_t*, size_t, int, uint32_t*);
void reference_pack(const uint8_t*, size_t, int, uint32_t*);
static void require(bool ok) { if (!ok) { puts("FAIL"); std::exit(1); } }
static std::vector<uint8_t> oracle(const uint8_t* q, size_t n, int bits) {
  std::vector<uint8_t> out(n * bits / 8);
  for (size_t j=0; j<n; ++j)
    for (int bit=0; bit<bits; ++bit)
      if (q[j] & (1u << bit)) out[(j*bits+bit)/8] |= 1u << ((j*bits+bit)%8);
  return out;
}
static size_t cases=0;
static void check(const std::vector<uint8_t>& source, size_t start, size_t n, int bits) {
  const auto copy=source;
  auto expected=oracle(source.data()+start,n,bits);
  std::vector<uint32_t> out(n*bits/32+2,0xabababab);
  ni::gguf::pack_codes(source.data()+start,n,bits,out.data()+1);
  require(out.front()==0xabababab && out.back()==0xabababab);
  require(expected.empty() || !memcmp(out.data()+1,expected.data(),expected.size()));
  require(copy==source); ++cases;
}
int main(int argc,char**argv) {
  std::mt19937 random(34797);
  std::vector<uint8_t> q(32);
  for(size_t lane=0;lane<32;++lane) for(int v=0;v<256;++v) {
    std::fill(q.begin(),q.end(),0);q[lane]=uint8_t(v);check(q,0,32,5);
  }
  for(int bits: {2,3,4,5,6,8}) for(size_t n: {0,32,64,96,256,512,2560,3136,4096,8192})
    for(size_t alignment=0;alignment<16;++alignment) {
      q.resize(n+alignment+16);for(auto& x:q)x=uint8_t(random());check(q,alignment,n,bits);
    }
  // The legacy path intentionally zeroes the output before reading overlapping source bytes.
  for(size_t n: {32,64,96,256,512}) for(size_t src=32;src<96;++src) for(size_t dst=0;dst<128;dst+=4) {
    std::vector<uint32_t> a(512),b;for(auto& x:a)x=random();b=a;
    reference_pack(reinterpret_cast<uint8_t*>(a.data())+src,n,5,a.data()+dst/4);
    ni::gguf::pack_codes(reinterpret_cast<uint8_t*>(b.data())+src,n,5,b.data()+dst/4);
    require(a==b);++cases;
  }
  printf("{\"oracle_and_overlap_cases\":%zu,\"result\":\"PASS\"}\n",cases);
  if(argc>1 && !strcmp(argv[1],"--check-only")) return 0;
  for(size_t n: {size_t(2560),size_t(3136),size_t(4096),size_t(8192)}) {
    const size_t rows=64;
    std::vector<float> x(n*rows);
    for(auto& f:x)f=(int(random()%20001)-10000)/5000.0f;
    std::vector<uint8_t> codes(n*rows);
    std::vector<uint16_t> sc(n*rows/64),bi(sc.size());
    std::vector<uint32_t> a(n*rows*5/32),b(a.size());
    ni::gguf::quantize(x.data(),x.size(),5,64,codes.data(),sc.data(),bi.data());
    reference_pack(codes.data(),codes.size(),5,a.data());
    ni::gguf::pack_codes(codes.data(),codes.size(),5,b.data());require(a==b);
    const auto stableCodes=codes;const auto stableSc=sc;const auto stableBi=bi;
    for(bool full: {false,true}) for(int leg=0;leg<8;++leg) {
      Pack pack=leg%2?ni::gguf::pack_codes:reference_pack;
      const int passes=full?4:256;
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
