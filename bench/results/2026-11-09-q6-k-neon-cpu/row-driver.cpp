#include "gguf/gguf.h"
#include "gguf/quant.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
int main(int argc,char**argv) {
 if(argc!=4)return 2;
 std::ifstream f(argv[1],std::ios::binary);std::vector<uint8_t>b((std::istreambuf_iterator<char>(f)),{});
 if(b.size()!=2150400)return 3;
 std::vector<float>y(1024*2560);
 for(int warm=0;warm<64;++warm)for(size_t row=0;row<1024;++row)ni::gguf::dequantize(ni::gguf::kQ6_K,b.data()+row*2100,y.data()+row*2560,2560);
 std::ifstream ef(argv[3],std::ios::binary);std::vector<char>expected((std::istreambuf_iterator<char>(ef)),{});
 if(expected.size()!=y.size()*sizeof(float))return 5;
 double ns=0;
 for(int pass=0;pass<256;++pass) {
  auto t=std::chrono::steady_clock::now();
  for(size_t row=0;row<1024;++row)ni::gguf::dequantize(ni::gguf::kQ6_K,b.data()+row*2100,y.data()+row*2560,2560);
  auto e=std::chrono::steady_clock::now();ns+=std::chrono::duration<double,std::nano>(e-t).count();
  if(memcmp(y.data(),expected.data(),expected.size()))return 6;
 }for(float x:y)if(!std::isfinite(x))return 4;
 std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<const char*>(y.data()),y.size()*sizeof(float));
 printf("%.9f\n",ns/(256*1024));
}
