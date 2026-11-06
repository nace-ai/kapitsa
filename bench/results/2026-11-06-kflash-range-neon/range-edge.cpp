#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <vector>
namespace ni {
void reference(const std::vector<const uint16_t*>&,uint32_t,int,bool,uint32_t*,uint16_t*,uint16_t*);
void candidate(const std::vector<const uint16_t*>&,uint32_t,int,bool,uint32_t*,uint16_t*,uint16_t*);
}
int main() {
  size_t count=0;
  for(int mode : {FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
    std::fesetround(mode);
    for(int bits : {4,8}) for(bool even : {false,true}) {
      std::vector<uint16_t> x(64); std::vector<const uint16_t*> rp{x.data()};
      std::vector<uint32_t> aw(2*bits),bw(2*bits);uint16_t as,ab,bs,bb;
      auto check=[&](const char* kind,uint32_t code) {
        ni::reference(rp,64,bits,even,aw.data(),&as,&ab);
        ni::candidate(rp,64,bits,even,bw.data(),&bs,&bb);
        if(aw!=bw || as!=bs || ab!=bb) { printf("FAIL %s code%04x mode%d bits%d even%d S%04x/%04x B%04x/%04x\n",kind,code,mode,bits,even,as,bs,ab,bb); return false; }
        ++count;return true;
      };
      for(uint32_t u=0;u<65536;++u) {
        std::fill(x.begin(),x.end(),uint16_t(u));
        if(!check("constant",u)) return 1;
        for(size_t i=0;i<x.size();++i) x[i]=uint16_t(0x3e00+(i%16)*17+((i&1)?0x8000:0));
        x[u%64]=uint16_t(u);
        if(!check("in-group",u)) return 1;
      }
      for(unsigned first=0;first<64;++first) for(unsigned second=0;second<64;++second) if(first!=second) {
        std::fill(x.begin(),x.end(),uint16_t(0x3f80)); x[first]=0x0000; x[second]=0x8000;
        if(!check("zero-ties",first*64+second)) return 1;
      }
    }
  }
  std::fesetround(FE_TONEAREST);
  printf("PASS %zu W/S/B comparisons: every BF16 pattern constant/in mixed groups; zero ties at every lane; 4 fenv modes; 4/8-bit and both rounding modes\n",count);
}
