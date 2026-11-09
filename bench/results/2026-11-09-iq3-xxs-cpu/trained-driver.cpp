#include "gguf/gguf.h"
#include "gguf/quant.h"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <iterator>
int main(int argc,char** argv){
 if(argc!=4)return 2;
 constexpr size_t rows=1024,k=1024,bytes=rows*k/256*98;
 std::ifstream input_file(argv[1],std::ios::binary);
 std::vector<uint8_t> input((std::istreambuf_iterator<char>(input_file)),{}),saved=input;
 if(input.size()!=bytes)return 3;
 std::vector<float> storage(rows*k+8,1234.0f);float* output=storage.data()+4;
 auto decode=[&](){for(size_t row=0;row<rows;++row)ni::gguf::dequantize(ni::gguf::kIQ3_XXS,input.data()+row*k/256*98,output+row*k,k);};
 decode();
 std::ifstream expected_file(argv[2],std::ios::binary);
 std::vector<char> gold((std::istreambuf_iterator<char>(expected_file)),{});
 if(gold.empty()){
  std::ofstream out(argv[2],std::ios::binary|std::ios::out);if(!out)return 4;
  out.write(reinterpret_cast<char*>(output),rows*k*sizeof(float));out.close();
  gold.assign(reinterpret_cast<char*>(output),reinterpret_cast<char*>(output)+rows*k*sizeof(float));
 }
 if(gold.size()!=rows*k*sizeof(float))return 5;
 printf("{\"ns_per_row\":[");
 for(int pass=0;pass<16;++pass){
  auto start=std::chrono::steady_clock::now();decode();auto end=std::chrono::steady_clock::now();
  if(memcmp(gold.data(),output,gold.size())||input!=saved)return 6;
  for(size_t i=0;i<rows*k;++i)if(!std::isfinite(output[i]))return 7;
  for(size_t i=0;i<4;++i)if(storage[i]!=1234.0f||storage[rows*k+4+i]!=1234.0f)return 8;
  printf("%s%.6f",pass?",":"",std::chrono::duration<double,std::nano>(end-start).count()/rows);
 }
 std::ofstream out(argv[3],std::ios::binary|std::ios::out);if(!out)return 9;
 out.write(reinterpret_cast<char*>(output),rows*k*sizeof(float));out.close();
 printf("],\"passes\":16,\"complete_bytes_per_pass\":4194304,\"finite\":true,\"input_unchanged\":true,\"canary\":true,\"equal_to_reference\":true}\n");return 0;
}
