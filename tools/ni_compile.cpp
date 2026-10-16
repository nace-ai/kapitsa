// Compiles the embedded Metal kernel library (and optionally creates pipelines for the given
// kernel names) without running anything on the GPU. Exit code 0 = all good.
//   ni_compile [kernel ...]
// NI_ARCHIVE_DIR=<dir>: also compiles each kernel to machine code (<dir>/<kernel>.bin, a Metal
// binary archive) and prints its code size, per-thread stack (private memory) and GPR count.
// Stack > 0 means arrays or spilled registers live in memory: v1-v10 of the prefill attention
// kept 6-13 KB of fragments there. The two numbers come from the undocumented compiler metadata
// (__GPU_METADATA) found by probing kernels with known register use; "?" if the layout differs.
#include "metal/gpu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
using Bytes = std::vector<uint8_t>;
uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint32_t le32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
uint64_t le64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

// Section (seg, sect) of the 64-bit Mach-O image at m[0..n).
bool macho_section(const uint8_t* m, size_t n, const char* seg, const char* sect, const uint8_t** out, size_t* len) {
  if (n < 32) return false;
  uint32_t ncmds = le32(m + 16);
  size_t off = 32;
  for (uint32_t i = 0; i < ncmds && off + 8 <= n; ++i) {
    uint32_t cmd = le32(m + off), csz = le32(m + off + 4);
    if (cmd == 0x19 && off + 72 <= n) {  // LC_SEGMENT_64
      uint32_t nsect = le32(m + off + 64);
      for (uint32_t s = 0; s < nsect; ++s) {
        const uint8_t* sd = m + off + 72 + 80 * s;
        if (sd + 80 > m + n) return false;
        if (strncmp((const char*)sd, sect, 16) == 0 && strncmp((const char*)sd + 16, seg, 16) == 0) {
          uint64_t size = le64(sd + 40);
          uint32_t soff = le32(sd + 48);
          if (soff + size > n) return false;
          *out = m + soff;
          *len = size;
          return true;
        }
      }
    }
    off += csz;
  }
  return false;
}

// -> code bytes, stack bytes, GPRs (-1 when not found)
void kernel_stats(const Bytes& b, long* text, long* stack, long* gprs) {
  *text = *stack = *gprs = -1;
  if (b.size() < 8) return;
  uint32_t nfat = be32(b.data() + 4);
  for (uint32_t i = 0; i < nfat && 8 + 20 * (i + 1) <= b.size(); ++i) {
    const uint8_t* e = b.data() + 8 + 20 * i;
    if (be32(e) != 0x01000013) continue;  // applegpu slice
    uint32_t off = be32(e + 8), size = be32(e + 12);
    if (off + size > b.size()) return;
    const uint8_t *inner, *txt, *md;
    size_t ninner, ntxt, nmd;
    if (!macho_section(b.data() + off, size, "__TEXT", "__compute", &inner, &ninner)) return;
    if (macho_section(inner, ninner, "__TEXT", "__text", &txt, &ntxt)) *text = long(ntxt);
    if (!macho_section(inner, ninner, "__GPU_METADATA", "__compute", &md, &nmd)) return;
    std::vector<uint32_t> w(nmd / 4);
    for (size_t k = 0; k < w.size(); ++k) w[k] = le32(md + 4 * k);
    // A flags word 0x01{01|03}xxxx is followed by the GPR count, or by the stack size (which also
    // appears in the few words before the flags) and then the GPR count. Without a stack the word
    // after the GPR count is 0 or 1, so a stack needs a plausible GPR count (>= 4) after it: a
    // 32-GPR kernel whose count equals the 0x20 word before the flags is not "32 B stack, 1 GPR".
    for (size_t k = 4; k + 2 < w.size(); ++k) {
      uint32_t f = w[k];
      if (f >> 24 != 1 || (((f >> 16) & 0xff) != 1 && ((f >> 16) & 0xff) != 3) || f == 0x01000000 || (f & 0xff) > 3)
        continue;
      uint32_t c = w[k + 1];
      bool seen = c > 0 && (w[k - 1] == c || w[k - 2] == c || w[k - 3] == c || w[k - 4] == c);
      if (seen && w[k + 2] >= 4 && w[k + 2] <= 256) { *stack = c; *gprs = w[k + 2]; return; }
      if (c <= 256) { *stack = 0; *gprs = c; return; }
    }
    return;
  }
}
}  // namespace

int main(int argc, char** argv) {
  ni::Gpu& g = ni::Gpu::get();
  g.compile();
  const char* adir = getenv("NI_ARCHIVE_DIR");
  for (int i = 1; i < argc; ++i) {
    ni::Pipeline p = g.pipeline(argv[i]);
    printf("%-36s ok (max threads %u)", argv[i], p.max_threads);
    if (adir) {
      std::string path = std::string(adir) + "/" + argv[i] + ".bin";
      std::ifstream f;
      if (g.archive(argv[i], path) && (f.open(path, std::ios::binary), f)) {
        Bytes b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        long text, stack, gprs;
        kernel_stats(b, &text, &stack, &gprs);
        if (stack >= 0) printf("  code %6ld B  stack %5ld B  gprs %3ld", text, stack, gprs);
        else printf("  code %6ld B  stack ? gprs ?", text);
      } else {
        printf("  (archive failed)");
      }
    }
    printf("\n");
  }
  printf("kernels compiled\n");
  return 0;
}
