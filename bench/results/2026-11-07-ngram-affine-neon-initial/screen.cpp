#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using Fn=void(*)(const uint32_t*,const uint16_t*,const uint16_t*,float*,int,int,int);
extern "C" void original(const uint32_t*,const uint16_t*,const uint16_t*,float*,int,int,int);
extern "C" void vector(const uint32_t*,const uint16_t*,const uint16_t*,float*,int,int,int);
int main(){
 std::mt19937 rng(123);size_t checks=0;
 for(int mode:{FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}){
  std::fesetround(mode);
  for(int d:{8,32,40,64,160,256,640})for(int gs:{8,16,32,64,128}){
   std::vector<uint32_t> q((d+7)/8);std::vector<uint16_t> s((d+gs-1)/gs),b(s.size());
   std::vector<float> a(d+2,12345.25f),z=a;
   for(unsigned round=0;round<65536;round++){
    for(size_t i=0;i<q.size();++i) q[i]=(round<16?round*0x11111111u:rng());
    for(size_t i=0;i<s.size();++i){s[i]=uint16_t(round);b[i]=uint16_t((round*131+int(i)*12345)&65535);}
    original(q.data(),s.data(),b.data(),a.data()+1,4,gs,d);
    vector(q.data(),s.data(),b.data(),z.data()+1,4,gs,d);
    for(int i=0;i<d+2;++i)if(std::bit_cast<uint32_t>(a[i])!=std::bit_cast<uint32_t>(z[i])){
     printf("MISMATCH mode=%d dim=%d gs=%d coeff=%u i=%d base=%08x new=%08x\n",mode,d,gs,round,i,std::bit_cast<uint32_t>(a[i]),std::bit_cast<uint32_t>(z[i]));return 1;
    }
    checks+=d;
   }
  }
 }
 std::fesetround(FE_TONEAREST);printf("PASS %zu values including guards, 4 rounding modes\n",checks);
 for(int d:{64,160,256,640})for(int gs:{32,64}){
  std::vector<uint32_t> q((d+7)/8);std::vector<uint16_t>s((d+gs-1)/gs,0x3b80),b(s.size(),0xbe80);std::vector<float>out(d);
  for(auto &x:q)x=rng();
  for(int k=0;k<4;++k){Fn f=k%2?vector:original;auto t=std::chrono::steady_clock::now();for(int j=0;j<200000;++j)f(q.data(),s.data(),b.data(),out.data(),4,gs,d);double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()/200000;printf("dim=%d gs=%d round=%d %s us=%.6f check=%g\n",d,gs,k,k%2?"B":"A",us,out[0]);}
 }
}
