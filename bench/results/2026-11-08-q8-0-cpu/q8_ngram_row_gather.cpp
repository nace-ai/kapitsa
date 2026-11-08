// CPU composition check: original Q8_0 tensor bytes in a synthetic GGUF table.
// This exercises NgramTable row/gather, not a trained PLE model, GPU or timing.
#include "model/ngram.h"
#include "gguf/gguf.h"
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
constexpr int ROWS = 1024;
float half_value(uint16_t h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  if (e == 31) throw std::runtime_error("Nonfinite scale in original Q8_0 fixture");
  const float mag = std::ldexp(float(e ? 1024 + m : m), e ? e - 25 : -24);
  return h & 0x8000 ? -mag : mag;
}
std::vector<uint8_t> read(const char* p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open original fixture");
  return {std::istreambuf_iterator<char>(f), {}};
}
void write_all(int fd, const uint8_t* p, size_t n) {
  while (n) {
    const auto r = ::write(fd, p, n);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) throw std::runtime_error("Cannot finish synthetic table");
    p += r; n -= size_t(r);
  }
}
void wrap(const char* path, const std::vector<uint8_t>& raw, int K) {
  std::vector<uint8_t> h;
  auto put = [&](const void* p, size_t n) { auto q=static_cast<const uint8_t*>(p); h.insert(h.end(),q,q+n); };
  auto u32=[&](uint32_t v){ put(&v,4); };
  auto u64=[&](uint64_t v){ put(&v,8); };
  auto str=[&](const char* v){ u64(std::strlen(v)); put(v,std::strlen(v)); };
  put("GGUF",4); u32(3); u64(1); u64(1);
  str("general.architecture"); u32(8); str("qwen4exp");
  str("per_layer_token_embd.weight"); u32(2); u64(K); u64(ROWS); u32(ni::gguf::kQ8_0); u64(0);
  h.resize((h.size()+31)/32*32,0);
  const int fd=::open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
  if (fd < 0) throw std::runtime_error("Synthetic table must be new");
  try {
    write_all(fd,h.data(),h.size()); write_all(fd,raw.data(),raw.size());
    if (::fsync(fd)) throw std::runtime_error("Cannot sync synthetic table");
  } catch (...) { ::close(fd); throw; }
  if (::close(fd)) throw std::runtime_error("Cannot close synthetic table");
}
void guards(const std::vector<float>& values, size_t used) {
  const auto* p=reinterpret_cast<const uint8_t*>(values.data());
  for (size_t i=0;i<16;++i) if (p[i]!=0xa5 || p[16+used*4+i]!=0xa5)
    throw std::runtime_error("Row/gather output guard overwritten");
}
}

int main(int argc,char** argv) try {
  if (argc!=4) throw std::runtime_error("Usage: q8-ngram-row-gather ORIGINAL_Q8_ROWS NEW_SYNTHETIC_GGUF WIDTH");
  const int K=std::stoi(argv[3]);
  if (K!=2560 && K!=4096) throw std::runtime_error("Only the two pinned original Q8_0 fixture widths are accepted");
  const int RB=(K/32)*34;
  const auto input=read(argv[1]);
  if (input.size()!=size_t(ROWS)*RB) throw std::runtime_error("Original Q8_0 fixture size mismatch");
  std::vector<float> oracle(size_t(ROWS)*K);
  for (int r=0;r<ROWS;++r) for (int b=0;b<K/32;++b) {
    const auto* p=input.data()+size_t(r)*RB+b*34;
    const float d=half_value(uint16_t(p[0])|(uint16_t(p[1])<<8));
    for (int j=0;j<32;++j) {
      const int q=p[2+j]<128 ? int(p[2+j]) : int(p[2+j])-256;
      oracle[size_t(r)*K+b*32+j]=d*float(q);
    }
  }
  wrap(argv[2],input,K);
  ni::NgramTable table;
  table.open_gguf(argv[2],"per_layer_token_embd.weight",K);
  if (!table.ok() || table.dim()!=K || table.rows()!=ROWS || table.describe().find("Q8_0")==std::string::npos)
    throw std::runtime_error("Synthetic Q8_0 table identity/shape mismatch");
  std::vector<float> row(K+8);
  for (int r=0;r<ROWS;++r) {
    std::memset(row.data(),0xa5,row.size()*sizeof(float));
    table.row(r,row.data()+4); guards(row,K);
    if (std::memcmp(row.data()+4,oracle.data()+size_t(r)*K,K*4)) throw std::runtime_error("Row independent oracle byte mismatch");
  }
  size_t gathered=0;
  for (auto [T,H]: {std::pair{1,16},{15,16},{16,16},{17,16},{65,16},{1024,1}}) {
    const size_t n=size_t(T)*H;
    std::vector<int64_t> ids(n);
    for (size_t i=0;i<n;++i) ids[i]=(i*37+(i%3?19:0))%ROWS;
    std::vector<float> got(n*K+8);
    std::memset(got.data(),0xa5,got.size()*sizeof(float));
    table.gather(ids.data(),T,H,got.data()+4); guards(got,n*K);
    for (size_t i=0;i<n;++i) if (std::memcmp(got.data()+4+i*K,oracle.data()+ids[i]*K,K*4))
      throw std::runtime_error("Gather independent oracle byte mismatch");
    std::printf("gather width=%d T=%d heads=%d rows=%zu full_bytes=%zu PASS\n",K,T,H,n,n*K*4);
    gathered+=n;
  }
  if (read(argv[1])!=input) throw std::runtime_error("Original Q8_0 input changed");
  const char* pre=std::getenv("NI_PLE_PREFETCH");
  std::printf("PASS width=%d original matrix %d rows + %zu gathered rows; complete independent oracle bytes and guards; "
              "synthetic Q8_0 wrapper; NI_PLE_PREFETCH=%s; no trained PLE/model/GPU/performance claim\n",
              K,ROWS,gathered,pre ? pre : "unset(default)");
  return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
