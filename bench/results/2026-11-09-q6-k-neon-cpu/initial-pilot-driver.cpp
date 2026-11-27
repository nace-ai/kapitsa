#include "gguf/gguf.h"
#include "gguf/quant.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cmath>
int main(int argc,char**argv) {
 if(argc!=3)return 2;
 std::ifstream f(argv[1],std::ios::binary);std::vector<uint8_t>b((std::istreambuf_iterator<char>(f)),{});
 if(b.size()!=2150400)return 3;
 std::vector<float>y(1024*2560);
 for(int warm=0;warm<4;++warm)for(size_t row=0;row<1024;++row)ni::gguf::dequantize(ni::gguf::kQ6_K,b.data()+row*2100,y.data()+row*2560,2560);
 auto t=std::chrono::steady_clock::now();
 for(int pass=0;pass<16;++pass)for(size_t row=0;row<1024;++row)ni::gguf::dequantize(ni::gguf::kQ6_K,b.data()+row*2100,y.data()+row*2560,2560);
 auto e=std::chrono::steady_clock::now();for(float x:y)if(!std::isfinite(x))return 4;
 std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<const char*>(y.data()),y.size()*sizeof(float));
 printf("%.9f\n",std::chrono::duration<double,std::nano>(e-t).count()/(16*1024));
}
