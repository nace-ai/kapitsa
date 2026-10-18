// Image sources, fetching and decoding (see image.h).
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#include <CommonCrypto/CommonDigest.h>

#include "hub/http.h"
#include "util/log.h"
#include "vision/image.h"

#include <algorithm>
#include <cstring>

namespace ni::vision {

using ojson = nlohmann::ordered_json;

FetchLimits fetch_limits() {
  FetchLimits l;
  if (const char* v = getenv("NI_IMAGE_MAX_MB"); v && atof(v) > 0) l.max_bytes = size_t(atof(v) * (1 << 20));
  if (const char* v = getenv("NI_IMAGE_TIMEOUT_S"); v && atof(v) > 0) l.timeout_s = atof(v);
  return l;
}

// ---- image parts ----

static bool typed(const ojson& it, const char* type) {
  const auto t = it.find("type");
  return t != it.end() && t->is_string() && t->get_ref<const std::string&>() == type;
}

// The template's test (`'image' in item or 'image_url' in item or item.type == 'image'`, videos likewise).
static bool is_image_part(const ojson& it) {
  if (it.is_string()) {
    const std::string& s = it.get_ref<const std::string&>();
    return s.find("image") != std::string::npos;  // `'image' in item` on a string is a substring test
  }
  return it.is_object() && (it.contains("image") || it.contains("image_url") || typed(it, "image"));
}
static bool is_video_part(const ojson& it) {
  if (it.is_string()) return it.get_ref<const std::string&>().find("video") != std::string::npos;
  return it.is_object() && (it.contains("video") || typed(it, "video"));
}

static Source source_of(const ojson& it) {
  Source s;
  if (!it.is_object()) throw Error("an image content part must be an object, e.g. {\"type\": \"image_url\", \"image_url\": {\"url\": ...}}");
  auto url_of = [](const ojson& v) -> std::string {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_object() && v.contains("url") && v["url"].is_string()) return v["url"].get<std::string>();
    return "";
  };
  if (it.contains("image_url")) s.url = url_of(it["image_url"]);
  else if (it.contains("image")) s.url = url_of(it["image"]);
  if (s.url.empty() && it.contains("source") && it["source"].is_object()) {  // Anthropic-style image block
    const ojson& src = it["source"];
    const bool b64 = typed(src, "base64"), url = typed(src, "url");
    const auto mt = src.find("media_type");
    if (b64 && src.contains("data") && src["data"].is_string())
      s.url = "data:" + (mt != src.end() && mt->is_string() ? mt->get<std::string>() : std::string("image")) + ";base64," +
              src["data"].get<std::string>();
    else if (url && src.contains("url") && src["url"].is_string()) s.url = src["url"].get<std::string>();
  }
  if (s.url.empty()) throw Error("an image content part has no image: expected {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,...\" or \"https://...\"}}");
  return s;
}

std::vector<Source> image_sources(const ojson& messages) {
  std::vector<Source> out;
  if (!messages.is_array()) return out;
  for (const ojson& m : messages) {
    if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
    for (const ojson& it : m["content"]) {
      if (is_image_part(it)) out.push_back(source_of(it));
      else if (is_video_part(it))
        throw Error("video input is not supported: Kapitsa takes images only (image_url parts with a data: or "
                    "http(s) URL)");
    }
  }
  return out;
}

ojson without_image_data(const ojson& messages) {
  if (!messages.is_array()) return messages;
  ojson out = messages;
  for (ojson& m : out) {
    if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
    for (ojson& it : m["content"])
      if (it.is_object() && is_image_part(it)) it = ojson{{"type", "image_url"}, {"image_url", {{"url", "image"}}}};
  }
  return out;
}

// ---- fetching ----

static std::string base64_decode(const char* p, size_t n) {
  static int8_t tab[256];
  static bool init = [] {
    memset(tab, -1, sizeof tab);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) tab[uint8_t(a[i])] = int8_t(i);
    tab[uint8_t('-')] = 62;  // URL-safe alphabet
    tab[uint8_t('_')] = 63;
    return true;
  }();
  (void)init;
  std::string out;
  out.reserve(n / 4 * 3 + 3);
  uint32_t acc = 0;
  int bits = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t c = uint8_t(p[i]);
    if (c == '=') break;
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    if (tab[c] < 0) throw Error("an image data: URL is not valid base64");
    acc = (acc << 6) | uint32_t(tab[c]);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(char((acc >> bits) & 0xFF));
    }
  }
  return out;
}

static std::string percent_decode(const char* p, size_t n) {
  std::string out;
  out.reserve(n);
  auto hexv = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
  for (size_t i = 0; i < n; ++i) {
    if (p[i] == '%' && i + 2 < n && hexv(p[i + 1]) >= 0 && hexv(p[i + 2]) >= 0) {
      out.push_back(char(hexv(p[i + 1]) * 16 + hexv(p[i + 2])));
      i += 2;
    } else {
      out.push_back(p[i]);
    }
  }
  return out;
}

static std::string mb(size_t b) {
  char s[32];
  snprintf(s, sizeof s, "%.1f MB", b / 1048576.0);
  return s;
}

std::string fetch(const Source& src, const FetchLimits& lim) {
  if (!src.bytes.empty()) return src.bytes;
  const std::string& u = src.url;
  auto starts = [&](const char* p) { return u.size() >= strlen(p) && strncasecmp(u.c_str(), p, strlen(p)) == 0; };
  if (starts("data:")) {
    const size_t comma = u.find(',');
    if (comma == std::string::npos) throw Error("an image data: URL has no ',' before its data");
    const std::string head = u.substr(5, comma - 5);
    const bool b64 = head.size() >= 7 && strcasecmp(head.c_str() + head.size() - 7, ";base64") == 0;
    std::string bytes = b64 ? base64_decode(u.data() + comma + 1, u.size() - comma - 1)
                            : percent_decode(u.data() + comma + 1, u.size() - comma - 1);
    if (bytes.size() > lim.max_bytes)
      throw Error("an image is " + mb(bytes.size()) + ", more than the " + mb(lim.max_bytes) + " Kapitsa takes (NI_IMAGE_MAX_MB)");
    if (bytes.empty()) throw Error("an image data: URL is empty");
    return bytes;
  }
  if (starts("http://") || starts("https://")) {
    hub::HttpClient http(lim.timeout_s);
    std::string body;
    const double t0 = now_ms();
    bool too_big = false, too_slow = false;
    const hub::HttpResult r = http.stream(u, {}, 0, 0, [&](const uint8_t* p, size_t n) {
      if (body.size() + n > lim.max_bytes) return !(too_big = true);
      if (now_ms() - t0 > lim.timeout_s * 1000) return !(too_slow = true);
      body.append(reinterpret_cast<const char*>(p), n);
      return true;
    });
    if (too_big) throw Error("the image at " + u + " is larger than " + mb(lim.max_bytes) + " (NI_IMAGE_MAX_MB)");
    if (too_slow) throw Error("the image at " + u + " took longer than " + std::to_string(int(lim.timeout_s)) + " s to download");
    if (!r.error.empty()) throw Error("can't download the image at " + u + ": " + r.error);
    if (r.status < 200 || r.status >= 300) throw Error("can't download the image at " + u + ": HTTP " + std::to_string(r.status));
    if (body.empty()) throw Error("the image at " + u + " is empty");
    return body;
  }
  throw Error("unsupported image URL \"" + u.substr(0, 40) + (u.size() > 40 ? "..." : "") +
              "\": Kapitsa takes data: URLs (base64) and http(s) URLs");
}

// ---- decoding ----

Digest sha256(const void* data, size_t len) {
  Digest d;
  CC_SHA256_CTX c;
  CC_SHA256_Init(&c);
  // CC_SHA256_Update takes a CC_LONG (32-bit) length
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (len > 0) {
    const size_t n = std::min(len, size_t(1) << 30);
    CC_SHA256_Update(&c, p, CC_LONG(n));
    p += n;
    len -= n;
  }
  CC_SHA256_Final(d.data(), &c);
  return d;
}

std::string hex(const Digest& d, size_t bytes) {
  static const char* x = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < std::min(bytes, d.size()); ++i) s += x[d[i] >> 4], s += x[d[i] & 15];
  return s;
}

// Raw pixels of the formats PNG and JPEG decode to, converted as PIL's convert("RGB") does: no color conversion,
// alpha dropped (premultiplied data is unpremultiplied first), 16-bit samples by their high byte, palettes looked
// up, gray replicated. False for anything else (the caller draws it instead).
static bool raw_rgb(CGImageRef img, std::vector<uint8_t>& out) {
  const size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
  const size_t bpc = CGImageGetBitsPerComponent(img), bpp = CGImageGetBitsPerPixel(img), bpr = CGImageGetBytesPerRow(img);
  const CGBitmapInfo info = CGImageGetBitmapInfo(img);
  if (info & kCGBitmapFloatComponents) return false;
  const CGImageAlphaInfo alpha = CGImageAlphaInfo(info & kCGBitmapAlphaInfoMask);
  const uint32_t order = info & kCGBitmapByteOrderMask;
  CGColorSpaceRef cs = CGImageGetColorSpace(img);
  if (!cs) return false;
  const CGColorSpaceModel model = CGColorSpaceGetModel(cs);
  if (bpc != 8 && bpc != 16) return false;
  const size_t B = bpc / 8;  // bytes per sample
  size_t ncomp = 0;          // samples per pixel
  std::vector<uint8_t> palette;
  if (model == kCGColorSpaceModelRGB) ncomp = 3;
  else if (model == kCGColorSpaceModelMonochrome) ncomp = 1;
  else if (model == kCGColorSpaceModelIndexed && bpc == 8) {
    CGColorSpaceRef base = CGColorSpaceGetBaseColorSpace(cs);
    if (!base || CGColorSpaceGetModel(base) != kCGColorSpaceModelRGB) return false;
    palette.assign(CGColorSpaceGetColorTableCount(cs) * 3, 0);
    CGColorSpaceGetColorTable(cs, palette.data());
    ncomp = 1;
  } else {
    return false;
  }
  const bool has_alpha = alpha != kCGImageAlphaNone && alpha != kCGImageAlphaOnly;
  if (alpha == kCGImageAlphaOnly) return false;
  const size_t spp = ncomp + (has_alpha ? 1 : 0);
  if (bpp != spp * bpc) return false;
  const bool alpha_first =
      alpha == kCGImageAlphaFirst || alpha == kCGImageAlphaPremultipliedFirst || alpha == kCGImageAlphaNoneSkipFirst;
  const bool premul = alpha == kCGImageAlphaPremultipliedFirst || alpha == kCGImageAlphaPremultipliedLast;
  // byte order: 32-bit little-endian pixels (8-bit samples) are stored reversed; 16-bit samples are big-endian
  // unless 16Little
  bool reversed = false, little16 = false;
  if (B == 1) {
    if (order == kCGBitmapByteOrder32Little) {
      if (spp != 4) return false;
      reversed = true;
    } else if (order == kCGBitmapByteOrder16Little || order == kCGBitmapByteOrder16Big) {
      return false;
    }
  } else {
    if (order == kCGBitmapByteOrder16Little) little16 = true;
    else if (order != kCGBitmapByteOrderDefault && order != kCGBitmapByteOrder16Big) return false;
  }
  CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(img));
  if (!data) return false;
  const uint8_t* base = CFDataGetBytePtr(data);
  if (size_t(CFDataGetLength(data)) < bpr * (h - 1) + w * spp * B) {
    CFRelease(data);
    return false;
  }
  out.resize(w * h * 3);
  const size_t ci = alpha_first ? 1 : 0;  // first color sample
  const size_t ai = alpha_first ? 0 : ncomp;
  for (size_t y = 0; y < h; ++y) {
    const uint8_t* row = base + y * bpr;
    uint8_t* o = out.data() + y * w * 3;
    for (size_t x = 0; x < w; ++x) {
      const uint8_t* px = row + x * spp * B;
      auto sample = [&](size_t i) -> uint32_t {
        if (reversed) i = spp - 1 - i;
        if (B == 1) return px[i];
        return little16 ? px[2 * i + 1] : px[2 * i];  // the high byte
      };
      uint32_t c[3];
      if (ncomp == 3) {
        c[0] = sample(ci), c[1] = sample(ci + 1), c[2] = sample(ci + 2);
      } else if (!palette.empty()) {
        const uint32_t k = std::min<uint32_t>(sample(ci), uint32_t(palette.size() / 3) - 1);
        c[0] = palette[3 * k], c[1] = palette[3 * k + 1], c[2] = palette[3 * k + 2];
      } else {
        c[0] = c[1] = c[2] = sample(ci);
      }
      if (premul) {
        const uint32_t a = sample(ai);
        for (uint32_t& v : c) v = a ? std::min<uint32_t>(255, (v * 255 + a / 2) / a) : 0;
      }
      o[3 * x] = uint8_t(c[0]), o[3 * x + 1] = uint8_t(c[1]), o[3 * x + 2] = uint8_t(c[2]);
    }
  }
  CFRelease(data);
  return true;
}

// Anything else (float, CMYK, Lab ...): drawn into an 8-bit RGB context (sRGB, opaque black background).
static void drawn_rgb(CGImageRef img, std::vector<uint8_t>& out) {
  const size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
  CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  std::vector<uint8_t> rgbx(w * h * 4);
  CGContextRef ctx = CGBitmapContextCreate(rgbx.data(), w, h, 8, w * 4, srgb,
                                           kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
  CGColorSpaceRelease(srgb);
  if (!ctx) throw Error("can't decode the image (unsupported pixel format)");
  CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
  CGContextRelease(ctx);
  out.resize(w * h * 3);
  for (size_t i = 0; i < w * h; ++i) memcpy(&out[3 * i], &rgbx[4 * i], 3);
}

// EXIF orientation 2-8 applied to an RGB image (as PIL's ImageOps.exif_transpose).
static void orient(std::vector<uint8_t>& rgb, int& w, int& h, int o) {
  if (o < 2 || o > 8) return;
  const bool swap = o >= 5;
  const int W = swap ? h : w, H = swap ? w : h;
  std::vector<uint8_t> out(rgb.size());
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int sx = x, sy = y;
      switch (o) {
        case 2: sx = w - 1 - x; break;                     // mirror left-right
        case 3: sx = w - 1 - x, sy = h - 1 - y; break;     // rotate 180
        case 4: sy = h - 1 - y; break;                     // mirror top-bottom
        case 5: sx = y, sy = x; break;                     // transpose
        case 6: sx = y, sy = h - 1 - x; break;             // rotate 90 clockwise
        case 7: sx = w - 1 - y, sy = h - 1 - x; break;     // transverse
        case 8: sx = w - 1 - y, sy = x; break;             // rotate 90 counter-clockwise
      }
      memcpy(&out[(size_t(y) * W + x) * 3], &rgb[(size_t(sy) * w + sx) * 3], 3);
    }
  rgb.swap(out);
  w = W, h = H;
}

Image decode(const std::string& bytes) {
  @autoreleasepool {
    CFDataRef data = CFDataCreateWithBytesNoCopy(nullptr, reinterpret_cast<const UInt8*>(bytes.data()),
                                                 CFIndex(bytes.size()), kCFAllocatorNull);
    CGImageSourceRef src = data ? CGImageSourceCreateWithData(data, nullptr) : nullptr;
    if (data) CFRelease(data);
    if (!src || CGImageSourceGetCount(src) < 1) {
      if (src) CFRelease(src);
      throw Error("can't decode the image: not a format macOS reads (PNG, JPEG, WebP, GIF, HEIC, TIFF, BMP ...)");
    }
    int orientation = 1;
    if (CFDictionaryRef props = CGImageSourceCopyPropertiesAtIndex(src, 0, nullptr)) {
      if (auto o = static_cast<CFNumberRef>(CFDictionaryGetValue(props, kCGImagePropertyOrientation)))
        CFNumberGetValue(o, kCFNumberIntType, &orientation);
      CFRelease(props);
    }
    NSDictionary* opts = @{(id)kCGImageSourceShouldCache : @NO, (id)kCGImageSourceShouldAllowFloat : @NO};
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, (__bridge CFDictionaryRef)opts);
    CFRelease(src);
    if (!img) throw Error("can't decode the image (damaged or unsupported file)");
    Image im;
    im.width = int(CGImageGetWidth(img));
    im.height = int(CGImageGetHeight(img));
    // a decompression bomb would take gigabytes before any resize
    if (im.width < 1 || im.height < 1 || double(im.width) * im.height > 100e6) {
      CGImageRelease(img);
      throw Error("the image is " + std::to_string(im.width) + " x " + std::to_string(im.height) +
                  " pixels: Kapitsa takes images of up to 100 megapixels");
    }
    try {
      if (!raw_rgb(img, im.rgb)) drawn_rgb(img, im.rgb);
    } catch (...) {
      CGImageRelease(img);
      throw;
    }
    CGImageRelease(img);
    orient(im.rgb, im.width, im.height, orientation);
    std::vector<uint8_t> head(8);
    memcpy(head.data(), &im.width, 4);
    memcpy(head.data() + 4, &im.height, 4);
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    CC_SHA256_Update(&c, head.data(), 8);
    for (size_t off = 0; off < im.rgb.size(); off += size_t(1) << 30)
      CC_SHA256_Update(&c, im.rgb.data() + off, CC_LONG(std::min(im.rgb.size() - off, size_t(1) << 30)));
    CC_SHA256_Final(im.digest.data(), &c);
    return im;
  }
}

}  // namespace ni::vision
