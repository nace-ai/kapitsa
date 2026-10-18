// Images for multimodal models, the part every model shares: the image parts of chat messages, fetching them
// (data: and http(s) URLs), decoding them (ImageIO) to 8-bit RGB, and their content digest (prefix reuse and
// checkpoints tell two images apart by it). Video is not supported.
#pragma once
#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ni::vision {

// A problem with a request's images, reported to the API client as is (bad URL, not an image, too large ...).
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

using Digest = std::array<uint8_t, 32>;  // SHA-256

// A decoded image: 8-bit RGB rows as PIL's Image.convert("RGB") gives them (the reference processors' input): no
// color management, alpha dropped (not composited), 16-bit samples by their high byte; the EXIF orientation applied.
struct Image {
  int width = 0, height = 0;
  std::vector<uint8_t> rgb;  // height x width x 3
  Digest digest{};           // SHA-256 of the dimensions and the RGB bytes: the same picture in any file format
};

struct FetchLimits {
  size_t max_bytes = size_t(32) << 20;  // encoded size cap (NI_IMAGE_MAX_MB)
  double timeout_s = 30;                // http(s): the whole download (NI_IMAGE_TIMEOUT_S)
};
FetchLimits fetch_limits();  // the defaults, with the environment's overrides

// One image of a request: where it comes from. A content part counts as an image exactly when the Qwen3.5-family
// chat templates render it as one: it has an "image" or "image_url" key, or "type": "image".
struct Source {
  std::string url;    // data: or http(s): URL
  std::string bytes;  // or the encoded bytes themselves (Anthropic-style {"source": {"type": "base64", ...}})
};
// The image parts of `messages` in the order a chat template renders them (messages in order, parts in order).
// Throws Error for video parts and for image parts without a usable source.
std::vector<Source> image_sources(const nlohmann::ordered_json& messages);
// `messages` with every image part's data replaced by a short stand-in: the template renders the same text without
// copying megabytes of base64 around.
nlohmann::ordered_json without_image_data(const nlohmann::ordered_json& messages);

// The encoded bytes of an image source: a data: URL (base64 or percent-encoded) or an http(s) download (size cap,
// timeout). Throws Error.
std::string fetch(const Source& src, const FetchLimits& lim);
// Decodes PNG, JPEG, WebP, GIF (first frame), HEIC, TIFF, BMP ... (whatever ImageIO reads). Throws Error.
Image decode(const std::string& bytes);
Digest sha256(const void* data, size_t len);
std::string hex(const Digest& d, size_t bytes = 6);  // short hex of a digest, for logs

}  // namespace ni::vision
