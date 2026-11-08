#include IMPORT_SOURCE
#include <chrono>
#include <iostream>

static uint16_t oracle_bf16(float x) {
  uint32_t u; memcpy(&u, &x, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  constexpr size_t K = 4096, N = 1024, HS = 128, NK = 16, RATIO = 2, ROW = K / 32 * 18;
  const int passes = std::stoi(argv[3]);
  if (passes < 1 || passes > 8192) return 3;
  std::ifstream file(argv[1], std::ios::binary);
  std::vector<uint8_t> input((std::istreambuf_iterator<char>(file)), {});
  if (input.size() != N * ROW) return 4;
  ni::gguf::Tensor tensor; tensor.type = ni::gguf::kQ4_0;
  ni::gguf::Lin l; l.src = &tensor; l.rows = N; l.K = K; l.srb = ROW;
  l.fmt = ni::gguf::k4g32; l.cperm_hs = HS; l.cperm_nk = NK;
  std::vector<uint8_t> packed(N*K/2), expected(packed.size());
  std::vector<uint16_t> scale(N*K/32), offset(scale.size()), es(scale.size()), eo(scale.size());
  for (size_t row = 0; row < N; ++row) {
    for (size_t h = 0; h < K/HS; ++h) {
      const size_t sh = (h % RATIO)*NK+h/RATIO;
      for (size_t j = 0; j < HS; ++j) {
        const size_t src = sh*HS+j, block = src/32, in = src%32;
        const uint8_t byte = input[row*ROW+block*18+2+in%16];
        const uint8_t code = in < 16 ? byte&15 : byte>>4;
        const size_t dst = row*K+h*HS+j;
        expected[dst/2] |= uint8_t(code << (4*(dst%2)));
      }
      for (size_t g = 0; g < HS/32; ++g) {
        uint16_t bits; memcpy(&bits, input.data()+row*ROW+(sh*HS/32+g)*18, 2);
        _Float16 half; memcpy(&half, &bits, 2); const float d = float(half);
        if (!std::isfinite(d)) return 5;
        const size_t dst = row*K/32+h*HS/32+g;
        es[dst]=oracle_bf16(d); eo[dst]=oracle_bf16(-8.0f*d);
      }
    }
  }
  auto run = [&] { ni::gguf::convert_rows(l,input.data(),0,0,N,packed.data(),scale.data(),offset.data()); };
  run();
  if (packed!=expected || scale!=es || offset!=eo) return 6;
  const auto start = std::chrono::steady_clock::now();
  for (int pass=0;pass<passes;++pass) run();
  const auto stop = std::chrono::steady_clock::now();
  if (packed!=expected || scale!=es || offset!=eo) return 7;
  std::ofstream out(argv[2],std::ios::binary);
  out.write(reinterpret_cast<const char*>(packed.data()),packed.size());
  out.write(reinterpret_cast<const char*>(scale.data()),scale.size()*2);
  out.write(reinterpret_cast<const char*>(offset.data()),offset.size()*2);
  if (!out) return 8;
  const double ns=std::chrono::duration<double,std::nano>(stop-start).count()/double(N*passes);
  std::cout<<"{\"rows\":"<<N<<",\"columns\":"<<K<<",\"passes\":"<<passes
           <<",\"ns_per_row\":"<<ns<<",\"complete_code_scale_offset_oracle\":true}"<<std::endl;
}
