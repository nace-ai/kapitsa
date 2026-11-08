#include IMPORT_SOURCE
#include <iostream>
#include <random>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ofstream output(argv[1],std::ios::binary);
  std::mt19937 gen(73419);
  size_t cases=0;
  for (uint32_t type : {ni::gguf::kQ4_0,ni::gguf::kQ4_1,ni::gguf::kQ4_K})
  for (size_t K : {256ul,512ul,1024ul,4096ul})
  for (size_t hs : {0ul,32ul,64ul,128ul})
  for (size_t nk : {1ul,2ul})
  for (size_t rows : {0ul,1ul,2ul,7ul,31ul})
  for (size_t shift : {0ul,3ul}) {
    const size_t s0=shift, r0=shift+2, r1=r0+rows;
    const size_t rowbytes=K/(type==ni::gguf::kQ4_K?256:32)*(type==ni::gguf::kQ4_K?144:type==ni::gguf::kQ4_0?18:20);
    std::vector<uint8_t> input((r1-s0)*rowbytes);
    for (auto& v:input) v=uint8_t(gen());
    ni::gguf::Tensor tensor; tensor.type=type;
    ni::gguf::Lin l; l.src=&tensor; l.rows=r1; l.K=K; l.srb=rowbytes; l.fmt=ni::gguf::k4g32;l.cperm_hs=hs;l.cperm_nk=nk;
    const size_t wb=rows*K/2, groups=rows*K/32;
    std::vector<uint8_t> w(wb+32,0xa7);
    std::vector<uint16_t> s(groups+16,0x4ca7),o(groups+16,0x8e29);
    ni::gguf::convert_rows(l,input.data(),s0,r0,r1,w.data()+16,s.data()+8,o.data()+8);
    for (size_t j=0;j<16;++j) if(w[j]!=0xa7||w[wb+16+j]!=0xa7)return 3;
    for (size_t j=0;j<8;++j) if(s[j]!=0x4ca7||s[groups+8+j]!=0x4ca7||o[j]!=0x8e29||o[groups+8+j]!=0x8e29)return 4;
    output.write(reinterpret_cast<const char*>(w.data()),w.size());
    output.write(reinterpret_cast<const char*>(s.data()),s.size()*2);
    output.write(reinterpret_cast<const char*>(o.data()),o.size()*2);
    ++cases;
  }
  if(!output)return 5;
  std::cout<<"{\"cases\":"<<cases<<",\"canaries\":true,\"input\":\"deterministic complete byte patterns; exceptional scales included\"}\n";
}
