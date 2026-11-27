#include "gguf_q4_codes_test.h"
#include <chrono>
#include <cmath>
#include <fstream>
int main(int argc,char**argv){
  if(argc!=3)return 2;
  constexpr size_t K=2560,rows=1024,passes=256,groups=K*rows/32;
  std::vector<uint8_t> input(groups*18),want(K*rows),got(K*rows);
  std::vector<float> ws(groups),wo(groups),gs(groups),go(groups);
  std::ifstream file(argv[1],std::ios::binary);
  if(!file.read(reinterpret_cast<char*>(input.data()),input.size())||file.peek()!=EOF)return 3;
  const auto original=input;
  for(size_t r=0;r<rows;++r)q4_codes_test::oracle(ni::gguf::kQ4_0,input.data()+r*K/32*18,want.data()+r*K,ws.data()+r*K/32,wo.data()+r*K/32,K);
  for(size_t i=0;i<groups;++i)if(!std::isfinite(ws[i])||!std::isfinite(wo[i]))return 5;
  double total=0;
  for(size_t p=0;p<64+passes;++p){
    auto a=std::chrono::steady_clock::now();
    for(size_t r=0;r<rows;++r)ni::gguf::decode4(ni::gguf::kQ4_0,input.data()+r*K/32*18,got.data()+r*K,gs.data()+r*K/32,go.data()+r*K/32,K);
    auto b=std::chrono::steady_clock::now();
    if(p>=64)total+=std::chrono::duration<double,std::nano>(b-a).count();
    if(got!=want||memcmp(gs.data(),ws.data(),groups*4)||memcmp(go.data(),wo.data(),groups*4)||input!=original)return 1;
  }
  std::ofstream out(argv[2],std::ios::binary);
  out.write(reinterpret_cast<const char*>(got.data()),got.size());
  out.write(reinterpret_cast<const char*>(gs.data()),groups*4);
  out.write(reinterpret_cast<const char*>(go.data()),groups*4);
  if(!out)return 4;
  printf("%.9f\n",total/(rows*passes));return 0;
}
