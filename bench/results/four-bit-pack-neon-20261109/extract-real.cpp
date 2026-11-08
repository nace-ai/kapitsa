#include "gguf/gguf.h"
#include "gguf/quant.h"
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdio>
int main(int argc,char**argv){
 if(argc!=4)return 2;
 ni::gguf::File f(argv[1]);
 for(const auto&t:f.tensors())if(t.type==ni::gguf::kQ4_0&&t.ne[0]==2560&&t.rows()>=1024){
  constexpr size_t rows=1024,K=2560;
  std::vector<uint8_t> raw(t.row_bytes()*rows),codes(K*rows);
  std::vector<float> scales(K/32),offsets(K/32);
  f.read(t,0,raw.data(),raw.size());
  for(size_t r=0;r<rows;++r){
   ni::gguf::decode4(t.type,raw.data()+r*t.row_bytes(),codes.data()+r*K,scales.data(),offsets.data(),K);
   for(float s:scales)if(!std::isfinite(s))return 1;
   for(float s:offsets)if(!std::isfinite(s))return 1;
  }
  std::ofstream(argv[2],std::ios::binary).write(reinterpret_cast<const char*>(raw.data()),raw.size());
  std::ofstream(argv[3],std::ios::binary).write(reinterpret_cast<const char*>(codes.data()),codes.size());
  nlohmann::json out={{"tensor",t.name},{"shape",t.ne},{"type",t.type},{"offset",t.offset},{"rows",rows},{"K",K},{"raw_bytes",raw.size()},{"code_bytes",codes.size()},{"all_scales_and_offsets_finite",true}};
  puts(out.dump(2).c_str());return 0;
 }return 3;
}
