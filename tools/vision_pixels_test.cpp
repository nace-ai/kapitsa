// CPU-only checks of the storage wrapped by the Qwen vision encoder, and of Inkling-2's processor (its size cap and
// transparency). No model or Metal device is needed.
#include "vision/encoder.h"
#include "vision/ink2_vit.h"
#include "vision/qwen_vit.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>

using namespace ni::vision;

namespace {
int fails = 0;
void check(bool ok, const char* what) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  fails += !ok;
}
}

int main() {
  const size_t page = pixel_page_size();
  bool aligned = true, bounded = true, copied = true, moved = true;
  for (size_t n : {size_t(1), size_t(16), page - 16, page - 1, page, page + 1, size_t(1376 * 864 * 3)}) {
    Prepared p;
    p.pixels.resize(n);
    aligned &= uintptr_t(p.pixels.data()) % page == 0;
    const size_t bytes = pixel_buffer_bytes(n);
    bounded &= bytes % page == 0 && bytes >= n + 16 && bytes < n + 16 + page;
    // Touch the promised tail, including the last mapped byte (also checked under AddressSanitizer).
    auto* storage = p.pixels.data();
    std::fill(storage, storage + bytes, uint8_t(0x5a));
    Prepared copy = p;
    copied &= copy.pixels.data() != storage && copy.pixels == p.pixels;
    copy.pixels[0] = 0;
    copied &= p.pixels[0] == 0x5a;
    Prepared next = std::move(p);
    moved &= next.pixels.data() == storage && next.pixels.size() == n;
    PixelBuffer().swap(next.pixels);
    moved &= next.pixels.empty() && next.pixels.capacity() == 0;
  }
  check(aligned, "pixel addresses are page aligned, including non-page-sized images");
  check(bounded, "mapped lengths cover the image and padding, without another full page");
  check(copied, "Prepared copies own their pixels independently");
  check(moved, "Prepared moves keep the address; metadata-only reset releases storage");

  PixelBuffer grown(3, 17);
  grown.reserve(page + 1);
  check(uintptr_t(grown.data()) % page == 0 && grown[0] == 17 && grown[2] == 17,
        "vector reallocation preserves alignment and contents");
  bool overflow = false;
  try {
    (void)pixel_buffer_bytes(std::numeric_limits<size_t>::max());
  } catch (const std::bad_array_new_length&) {
    overflow = true;
  }
  check(overflow, "page rounding rejects size overflow");

  // Identity, one-axis, two-pass, and Pillow's vertical-first tall-image branch all return wrappable storage.
  struct Shape { int w, h, ow, oh; };
  bool resize_ok = true;
  for (const auto s : {Shape{32, 32, 32, 32}, Shape{32, 32, 64, 32}, Shape{32, 32, 32, 64},
                       Shape{37, 53, 32, 64}, Shape{2, 300, 32, 64}}) {
    const std::vector<uint8_t> rgb(size_t(s.w) * s.h * 3, 73);
    const auto out = pil_resize_bicubic(rgb, s.w, s.h, s.ow, s.oh);
    resize_ok &= uintptr_t(out.data()) % page == 0 && out.size() == size_t(s.ow) * s.oh * 3;
    resize_ok &= std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 73; });
  }
  check(resize_ok, "every bicubic resize branch writes directly into aligned pixel storage");
  // Inkling-2: the final token grid (padding included) within the cap, neither side enlarged, the aspect kept
  {
    bool ok = true;
    const long cap = 1200000;  // 750 tokens of 40 x 40
    for (auto [w, h] : {std::pair{25, 65520}, {65520, 25}, {2880, 1800}, {1095, 1095}, {39, 39}, {1, 100000}, {4096, 2414}}) {
      const auto [ow, oh] = ink2_target_size(w, h, 40, cap);
      const long tokens = long((ow + 39) / 40) * ((oh + 39) / 40);
      const bool fits = tokens * 1600 <= cap, kept = ow <= w && oh <= h && ow >= 1 && oh >= 1;
      const bool whole = long((w + 39) / 40) * ((h + 39) / 40) * 1600 <= cap ? (ow == w && oh == h) : true;
      printf("     %d x %d -> %d x %d (%ld tokens)\n", w, h, ow, oh, tokens);
      ok &= fits && kept && whole;
    }
    const auto [w0, h0] = ink2_target_size(25, 65520, 40, 0);
    ok &= w0 == 25 && h0 == 65520;
    check(ok, "Inkling-2 size cap: final token grid within 750 tokens, no side enlarged, 0 keeps the image");
  }
  // Inkling-2: transparency composited onto the 8-pixel checkerboard as the reference does, and in the digest
  {
    const Ink2ViT enc("", Ink2ViTConfig{});
    Image im;
    im.width = 20, im.height = 10;
    im.rgb.assign(size_t(im.width) * im.height * 3, 0);  // black
    im.alpha.assign(size_t(im.width) * im.height, 0);    // transparent
    for (int x = 0; x < im.width; ++x) im.alpha[size_t(5) * im.width + x] = uint8_t(x * 13);  // a partial row
    im.alpha[0] = 255;  // one opaque black pixel
    const Prepared p = enc.prepare(im);
    bool ok = p.pixels.size() == im.rgb.size();
    for (int y = 0; ok && y < im.height; ++y)
      for (int x = 0; x < im.width; ++x) {
        const uint32_t a = im.alpha[size_t(y) * im.width + x], bg = ((y / 8 + x / 8) % 2 == 0) ? 204 : 153;
        const uint8_t want = uint8_t((bg * (255 - a) + 127) / 255);
        for (int c = 0; c < 3; ++c) ok &= p.pixels[(size_t(y) * im.width + x) * 3 + c] == want;
      }
    Image opaque = im;
    opaque.alpha.clear();
    const Prepared q = enc.prepare(opaque);
    ok &= std::all_of(q.pixels.begin(), q.pixels.end(), [](uint8_t v) { return v == 0; }) && q.digest != p.digest;
    check(ok, "Inkling-2 transparency: the checkerboard composite to the bit; transparent and opaque black differ");
  }
  return fails ? 1 : 0;
}
