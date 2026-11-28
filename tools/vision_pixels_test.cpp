// CPU-only checks of the storage wrapped by the Qwen vision encoder, of the decoder's alpha (a padding byte is not
// transparency), and of Inkling-2's processor (its size cap and transparency) and config checks. No model or Metal
// device is needed.
#include "vision/encoder.h"
#include "vision/ink2_vit.h"
#include "vision/qwen_vit.h"

#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
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
  // The decoder: a 32-bit BMP's fourth byte is padding (ImageIO: NoneSkipLast, RGBX ff 00 00 7f), so the picture is
  // opaque red with no alpha plane; a PNG with real alpha keeps it (review 6073856524 on #1419)
  {
    const uint8_t kRgbxBmp[] = {0x42, 0x4d, 0x3a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x28,
                                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x20, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x13, 0x0b, 0x00, 0x00, 0x13, 0x0b, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x7f};
    const uint8_t kRgbaPng[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
                                0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
                                0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60,
                                0x60, 0xf8, 0xdf, 0x00, 0x00, 0x02, 0x83, 0x01, 0x80, 0x43, 0xbb, 0xf8, 0x03, 0x00, 0x00,
                                0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
    const Image bmp = decode(std::string(reinterpret_cast<const char*>(kRgbxBmp), sizeof kRgbxBmp));
    const Image png = decode(std::string(reinterpret_cast<const char*>(kRgbaPng), sizeof kRgbaPng));
    printf("     BMP rgb %d %d %d, alpha plane %zu; PNG rgb %d %d %d, alpha %d\n", bmp.rgb[0], bmp.rgb[1], bmp.rgb[2],
           bmp.alpha.size(), png.rgb[0], png.rgb[1], png.rgb[2], png.alpha.empty() ? -1 : png.alpha[0]);
    check(bmp.width == 1 && bmp.rgb.size() == 3 && bmp.rgb[0] == 255 && bmp.rgb[1] == 0 && bmp.rgb[2] == 0 &&
              bmp.alpha.empty(),
          "decoder: an opaque 32-bit BMP (RGBX) is red with no alpha plane");
    check(png.rgb.size() == 3 && png.rgb[2] == 255 && png.alpha.size() == 1 && png.alpha[0] == 128,
          "decoder: an RGBA PNG keeps its alpha (128)");
    const Ink2ViT enc("", Ink2ViTConfig{});
    const Prepared p = enc.prepare(bmp);
    check(p.pixels.size() == 3 && p.pixels[0] == 255 && p.pixels[1] == 0 && p.pixels[2] == 0,
          "Inkling-2: the opaque BMP reaches the tower as red, not the checkerboard's mix");
  }
  // Inkling-2's config checks: a tower with a zero or negative patch size is refused before anything divides by it
  // (review 6073856524 on #1419); the same folder with a 20-pixel patch is taken
  {
    char root[] = "/tmp/ink2_supports_XXXXXX";
    const bool made = mkdtemp(root) != nullptr;
    const std::string dir = root;
    {  // the two tensors supports() looks for, empty
      const std::string header = R"({"model.visual.backbone.embeddings.patch_embedding.weight":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]},)"
                                 R"("model.visual.proj_out.weight":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]}})";
      std::ofstream f(dir + "/model.safetensors", std::ios::binary);
      const uint64_t n = header.size();
      f.write(reinterpret_cast<const char*>(&n), 8);
      f << header;
    }
    auto with_patch = [&](int patch) {
      std::ofstream(dir + "/config.json")
          << R"({"model_type":"inkling2_loop","vision_config":{"model_type":"inkling2_vision","native_patch_size":)"
          << patch << R"(,"adapter_merge_size":2,"hidden_size":1024,"num_attention_heads":16,"num_hidden_layers":23,)"
          << R"("text_hidden_size":2304,"projection_hidden_dim":4096,"posemb_grid_size":32}})";
      Ink2ViTConfig c;
      return Ink2ViT::supports(dir, &c);
    };
    const bool good = with_patch(20), zero = with_patch(0), negative = with_patch(-20);
    printf("     supports: patch 20 %d, patch 0 %d, patch -20 %d\n", good, zero, negative);
    check(made && good && !zero && !negative, "Inkling-2: a zero or negative patch size is refused, 20 is taken");
    unlink((dir + "/config.json").c_str());
    unlink((dir + "/model.safetensors").c_str());
    rmdir(root);
  }
  return fails ? 1 : 0;
}
