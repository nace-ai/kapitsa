#include "gguf/quant.h"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <iterator>
int main(int argc, char** argv) {
  if(argc != 3) return 2;
  constexpr size_t K=2560, rows=1024, passes=256;
  std::vector<uint8_t> input(K*rows), expected(K*rows/2);
  std::ifstream file(argv[1], std::ios::binary);
  if(!file.read(reinterpret_cast<char*>(input.data()), input.size()) || file.peek()!=EOF) return 3;
  const auto original=input;
  for(size_t i=0;i<input.size();i+=2)expected[i/2]=uint8_t((input[i]&15)|((input[i+1]&15)<<4));
  std::vector<uint32_t> output(K*rows/8);
  double total=0;
  for(size_t p=0;p<64+passes;++p){
    auto a=std::chrono::steady_clock::now();
    for(size_t row=0;row<rows;++row)ni::gguf::pack_codes(input.data()+row*K,K,4,output.data()+row*K/8);
    auto b=std::chrono::steady_clock::now();
    if(p>=64)total+=std::chrono::duration<double,std::nano>(b-a).count();
    if(memcmp(output.data(),expected.data(),expected.size()) || input!=original)return 1;
  }
  std::ofstream out(argv[2],std::ios::binary);
  if(!out.write(reinterpret_cast<const char*>(output.data()),expected.size())) return 4;
  printf("%.9f\n",total/(rows*passes));return 0;
}
