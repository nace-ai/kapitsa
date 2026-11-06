#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include "util/bf16.h"
namespace ni {
void reference(const std::vector<const uint16_t*>&,uint32_t,int,bool,uint32_t*,uint16_t*,uint16_t*);
void candidate(const std::vector<const uint16_t*>&,uint32_t,int,bool,uint32_t*,uint16_t*,uint16_t*);
}
int main(){
 std::mt19937 rng(953);size_t cases=0;
 for(int bits:{4,8}) for(bool even:{false,true}) for(unsigned N:{1,8,63,64,65}) for(unsigned K:{64,128,256,2560}) {
  std::vector<uint16_t> x(size_t(N)*K);std::vector<const uint16_t*> rows(N);
  for(unsigned i=0;i<N;++i) rows[i]=x.data()+size_t(i)*K;
  for(int pattern=0;pattern<20;++pattern){
   for(size_t i=0;i<x.size();++i){float v=0;
    if(pattern==1)v=-0.0f;else if(pattern==2)v=1;else if(pattern==3)v=-1;
    else if(pattern==4)v=float(i%64)/2;else if(pattern==5)v=-float(i%64)/2;
    else if(pattern>=6)v=std::ldexp(float(int(rng()%65537)-32768),pattern-23);
    x[i]=ni::f32_to_bf16(v);
   }
   std::vector<uint32_t>a(size_t(N)*K*bits/32),b(a.size());std::vector<uint16_t>as(size_t(N)*K/64),ab(as.size()),bs(as.size()),bb(as.size());
   ni::reference(rows,K,bits,even,a.data(),as.data(),ab.data());ni::candidate(rows,K,bits,even,b.data(),bs.data(),bb.data());
   if(a!=b||as!=bs||ab!=bb){printf("FAIL bits%d even%d N%u K%u pattern%d\n",bits,even,N,K,pattern);return 1;}++cases;
  }
 }
 printf("PASS %zu cases W/S/B bitwise\n",cases);
}
