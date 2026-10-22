#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <fstream>

namespace ni {

#include "tokenizer/unicode_tables.inc"
#include "tokenizer/unicode_tables_cased.inc"

namespace {

template <size_t N>
bool in_table(const uint32_t (&t)[N][2], uint32_t cp) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < t[mid][0]) hi = mid;
    else if (cp > t[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

inline bool is_L(uint32_t c) { return c < 0x80 ? ((c | 32) - 'a' < 26) : in_table(kLetter, c); }
inline bool is_M(uint32_t c) { return c >= 0x300 && in_table(kMark, c); }
inline bool is_N(uint32_t c) { return c < 0x80 ? (c - '0' < 10) : in_table(kNumber, c); }
inline bool is_space(uint32_t c) {  // Unicode White_Space
  return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

struct CP {
  uint32_t c;
  uint32_t off;  // byte offset in the UTF-8 string
};

std::vector<CP> to_cps(const std::string& s) {
  std::vector<CP> out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    unsigned char b = s[i];
    uint32_t c;
    int n;
    if (b < 0x80) { c = b; n = 1; }
    else if ((b >> 5) == 6 && i + 1 < s.size()) { c = b & 0x1F; n = 2; }
    else if ((b >> 4) == 14 && i + 2 < s.size()) { c = b & 0x0F; n = 3; }
    else if ((b >> 3) == 30 && i + 3 < s.size()) { c = b & 0x07; n = 4; }
    else { c = 0xFFFD; n = 1; }  // invalid byte: treat as one opaque code point
    for (int k = 1; k < n; ++k) c = (c << 6) | (s[i + k] & 0x3F);
    out.push_back({c, uint32_t(i)});
    i += n;
  }
  return out;
}

// Qwen3.5 pre-tokenizer:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
void pretokenize(const std::string& s, std::vector<std::string>& out) {
  std::vector<CP> cps = to_cps(s);
  const size_t n = cps.size();
  auto cp = [&](size_t i) -> uint32_t { return i < n ? cps[i].c : 0xFFFFFFFF; };
  auto LM = [&](size_t i) { return i < n && (is_L(cps[i].c) || is_M(cps[i].c)); };
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    uint32_t c = cps[i].c;
    // 1. contractions
    if (c == '\'' && i + 1 < n) {
      uint32_t a = cp(i + 1) | 32, b = i + 2 < n ? (cp(i + 2) | 32) : 0;
      if (a == 's' || a == 't' || a == 'm' || a == 'd') j = i + 2;
      else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) j = i + 3;
    }
    // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if (j == i) {
      size_t k = i;
      if (!is_nl(c) && !is_L(c) && !is_N(c) && LM(i + 1)) k = i + 1;
      if (LM(k)) {
        while (LM(k)) ++k;
        j = k;
      }
    }
    // 3. \p{N}
    if (j == i && is_N(c)) j = i + 1;
    // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    if (j == i) {
      size_t k = i;
      auto punct = [&](size_t q) {
        if (q >= n) return false;
        uint32_t x = cps[q].c;
        return !is_space(x) && !is_L(x) && !is_M(x) && !is_N(x);
      };
      if (c == ' ' && punct(i + 1)) k = i + 1;
      if (punct(k)) {
        while (punct(k)) ++k;
        while (k < n && is_nl(cps[k].c)) ++k;
        j = k;
      }
    }
    // 5-7. whitespace
    if (j == i && is_space(c)) {
      size_t k = i;
      while (k < n && is_space(cps[k].c)) ++k;  // maximal whitespace run [i, k)
      size_t last_nl = n;
      for (size_t q = i; q < k; ++q)
        if (is_nl(cps[q].c)) last_nl = q;
      if (last_nl != n) {
        j = last_nl + 1;                         // \s*[\r\n]+
      } else if (k < n && k - i > 1) {
        j = k - 1;                               // \s+(?!\S): leave one space for the next word
      } else {
        j = k;                                   // \s+
      }
    }
    if (j == i) j = i + 1;  // unreachable for valid input; never stall
    uint32_t b0 = cps[i].off, b1 = j < n ? cps[j].off : uint32_t(s.size());
    out.emplace_back(s.substr(b0, b1 - b0));
    i = j;
  }
}

inline bool is_Lut(uint32_t c) { return c < 0x80 ? (c - 'A' < 26) : in_table(kUpperTitle, c); }   // Lu, Lt
inline bool is_Ll(uint32_t c) { return c < 0x80 ? (c - 'a' < 26) : in_table(kLowerCase, c); }      // Ll
inline bool is_Lmo(uint32_t c) { return c >= 0x80 && in_table(kModOther, c); }                    // Lm, Lo

// Tekken pre-tokenizer (Mistral's tokenizer.json, Nemotron; transformers applies the same pattern with
// fix_mistral_regex):
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
// |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
// |\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Each alternative is matched as the regex engine does (first alternative that matches, greedy with
// backtracking): A* B+ backtracks A* until a B follows.
void pretokenize_tekken(const std::string& s, std::vector<std::string>& out) {
  std::vector<CP> cps = to_cps(s);
  const size_t n = cps.size();
  auto at = [&](size_t i) { return cps[i].c; };
  auto P = [&](size_t i) { return i < n && !is_nl(at(i)) && !is_L(at(i)) && !is_N(at(i)); };
  auto A = [&](size_t i) { return i < n && (is_Lut(at(i)) || is_Lmo(at(i)) || is_M(at(i))); };
  auto B = [&](size_t i) { return i < n && (is_Ll(at(i)) || is_Lmo(at(i)) || is_M(at(i))); };
  auto Q = [&](size_t i) { return i < n && !is_space(at(i)) && !is_L(at(i)) && !is_N(at(i)); };
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    // 1. P? A* B+
    for (int pre = P(i) ? 1 : 0; pre >= 0 && j == i; --pre) {
      const size_t j0 = i + size_t(pre);
      size_t a_end = j0;
      while (A(a_end)) ++a_end;
      for (size_t k = a_end + 1; k-- > j0;) {
        if (!B(k)) continue;
        size_t e = k;
        while (B(e)) ++e;
        j = e;
        break;
      }
    }
    // 2. P? A+ B*
    for (int pre = P(i) ? 1 : 0; pre >= 0 && j == i; --pre) {
      const size_t j0 = i + size_t(pre);
      size_t a_end = j0;
      while (A(a_end)) ++a_end;
      if (a_end == j0) continue;
      while (B(a_end)) ++a_end;
      j = a_end;
    }
    // 3. \p{N}
    if (j == i && is_N(at(i))) j = i + 1;
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
    if (j == i) {
      size_t k = at(i) == ' ' && Q(i + 1) ? i + 1 : i;
      if (Q(k)) {
        while (Q(k)) ++k;
        while (k < n && (is_nl(at(k)) || at(k) == '/')) ++k;
        j = k;
      }
    }
    // 5-7. whitespace, as in the Qwen split
    if (j == i && is_space(at(i))) {
      size_t k = i;
      while (k < n && is_space(at(k))) ++k;
      size_t last_nl = n;
      for (size_t q = i; q < k; ++q)
        if (is_nl(at(q))) last_nl = q;
      if (last_nl != n) j = last_nl + 1;
      else if (k < n && k - i > 1) j = k - 1;
      else j = k;
    }
    if (j == i) j = i + 1;
    const uint32_t b0 = cps[i].off, b1 = j < n ? cps[j].off : uint32_t(s.size());
    out.emplace_back(s.substr(b0, b1 - b0));
    i = j;
  }
}

// \p{P} and \p{S}: the code points whose category starts with P or S (Unicode 15.0.0, Python 3.12's unicodedata, the
// version of unicode_tables*.inc)
static const uint32_t kPunctSymbol[][2] = {
  {0x21, 0x2F}, {0x3A, 0x40}, {0x5B, 0x60}, {0x7B, 0x7E}, {0xA1, 0xA9}, {0xAB, 0xAC}, {0xAE, 0xB1}, {0xB4, 0xB4},
  {0xB6, 0xB8}, {0xBB, 0xBB}, {0xBF, 0xBF}, {0xD7, 0xD7}, {0xF7, 0xF7}, {0x2C2, 0x2C5}, {0x2D2, 0x2DF},
  {0x2E5, 0x2EB}, {0x2ED, 0x2ED}, {0x2EF, 0x2FF}, {0x375, 0x375}, {0x37E, 0x37E}, {0x384, 0x385}, {0x387, 0x387},
  {0x3F6, 0x3F6}, {0x482, 0x482}, {0x55A, 0x55F}, {0x589, 0x58A}, {0x58D, 0x58F}, {0x5BE, 0x5BE}, {0x5C0, 0x5C0},
  {0x5C3, 0x5C3}, {0x5C6, 0x5C6}, {0x5F3, 0x5F4}, {0x606, 0x60F}, {0x61B, 0x61B}, {0x61D, 0x61F}, {0x66A, 0x66D},
  {0x6D4, 0x6D4}, {0x6DE, 0x6DE}, {0x6E9, 0x6E9}, {0x6FD, 0x6FE}, {0x700, 0x70D}, {0x7F6, 0x7F9}, {0x7FE, 0x7FF},
  {0x830, 0x83E}, {0x85E, 0x85E}, {0x888, 0x888}, {0x964, 0x965}, {0x970, 0x970}, {0x9F2, 0x9F3}, {0x9FA, 0x9FB},
  {0x9FD, 0x9FD}, {0xA76, 0xA76}, {0xAF0, 0xAF1}, {0xB70, 0xB70}, {0xBF3, 0xBFA}, {0xC77, 0xC77}, {0xC7F, 0xC7F},
  {0xC84, 0xC84}, {0xD4F, 0xD4F}, {0xD79, 0xD79}, {0xDF4, 0xDF4}, {0xE3F, 0xE3F}, {0xE4F, 0xE4F}, {0xE5A, 0xE5B},
  {0xF01, 0xF17}, {0xF1A, 0xF1F}, {0xF34, 0xF34}, {0xF36, 0xF36}, {0xF38, 0xF38}, {0xF3A, 0xF3D}, {0xF85, 0xF85},
  {0xFBE, 0xFC5}, {0xFC7, 0xFCC}, {0xFCE, 0xFDA}, {0x104A, 0x104F}, {0x109E, 0x109F}, {0x10FB, 0x10FB},
  {0x1360, 0x1368}, {0x1390, 0x1399}, {0x1400, 0x1400}, {0x166D, 0x166E}, {0x169B, 0x169C}, {0x16EB, 0x16ED},
  {0x1735, 0x1736}, {0x17D4, 0x17D6}, {0x17D8, 0x17DB}, {0x1800, 0x180A}, {0x1940, 0x1940}, {0x1944, 0x1945},
  {0x19DE, 0x19FF}, {0x1A1E, 0x1A1F}, {0x1AA0, 0x1AA6}, {0x1AA8, 0x1AAD}, {0x1B5A, 0x1B6A}, {0x1B74, 0x1B7E},
  {0x1BFC, 0x1BFF}, {0x1C3B, 0x1C3F}, {0x1C7E, 0x1C7F}, {0x1CC0, 0x1CC7}, {0x1CD3, 0x1CD3}, {0x1FBD, 0x1FBD},
  {0x1FBF, 0x1FC1}, {0x1FCD, 0x1FCF}, {0x1FDD, 0x1FDF}, {0x1FED, 0x1FEF}, {0x1FFD, 0x1FFE}, {0x2010, 0x2027},
  {0x2030, 0x205E}, {0x207A, 0x207E}, {0x208A, 0x208E}, {0x20A0, 0x20C0}, {0x2100, 0x2101}, {0x2103, 0x2106},
  {0x2108, 0x2109}, {0x2114, 0x2114}, {0x2116, 0x2118}, {0x211E, 0x2123}, {0x2125, 0x2125}, {0x2127, 0x2127},
  {0x2129, 0x2129}, {0x212E, 0x212E}, {0x213A, 0x213B}, {0x2140, 0x2144}, {0x214A, 0x214D}, {0x214F, 0x214F},
  {0x218A, 0x218B}, {0x2190, 0x2426}, {0x2440, 0x244A}, {0x249C, 0x24E9}, {0x2500, 0x2775}, {0x2794, 0x2B73},
  {0x2B76, 0x2B95}, {0x2B97, 0x2BFF}, {0x2CE5, 0x2CEA}, {0x2CF9, 0x2CFC}, {0x2CFE, 0x2CFF}, {0x2D70, 0x2D70},
  {0x2E00, 0x2E2E}, {0x2E30, 0x2E5D}, {0x2E80, 0x2E99}, {0x2E9B, 0x2EF3}, {0x2F00, 0x2FD5}, {0x2FF0, 0x2FFB},
  {0x3001, 0x3004}, {0x3008, 0x3020}, {0x3030, 0x3030}, {0x3036, 0x3037}, {0x303D, 0x303F}, {0x309B, 0x309C},
  {0x30A0, 0x30A0}, {0x30FB, 0x30FB}, {0x3190, 0x3191}, {0x3196, 0x319F}, {0x31C0, 0x31E3}, {0x3200, 0x321E},
  {0x322A, 0x3247}, {0x3250, 0x3250}, {0x3260, 0x327F}, {0x328A, 0x32B0}, {0x32C0, 0x33FF}, {0x4DC0, 0x4DFF},
  {0xA490, 0xA4C6}, {0xA4FE, 0xA4FF}, {0xA60D, 0xA60F}, {0xA673, 0xA673}, {0xA67E, 0xA67E}, {0xA6F2, 0xA6F7},
  {0xA700, 0xA716}, {0xA720, 0xA721}, {0xA789, 0xA78A}, {0xA828, 0xA82B}, {0xA836, 0xA839}, {0xA874, 0xA877},
  {0xA8CE, 0xA8CF}, {0xA8F8, 0xA8FA}, {0xA8FC, 0xA8FC}, {0xA92E, 0xA92F}, {0xA95F, 0xA95F}, {0xA9C1, 0xA9CD},
  {0xA9DE, 0xA9DF}, {0xAA5C, 0xAA5F}, {0xAA77, 0xAA79}, {0xAADE, 0xAADF}, {0xAAF0, 0xAAF1}, {0xAB5B, 0xAB5B},
  {0xAB6A, 0xAB6B}, {0xABEB, 0xABEB}, {0xFB29, 0xFB29}, {0xFBB2, 0xFBC2}, {0xFD3E, 0xFD4F}, {0xFDCF, 0xFDCF},
  {0xFDFC, 0xFDFF}, {0xFE10, 0xFE19}, {0xFE30, 0xFE52}, {0xFE54, 0xFE66}, {0xFE68, 0xFE6B}, {0xFF01, 0xFF0F},
  {0xFF1A, 0xFF20}, {0xFF3B, 0xFF40}, {0xFF5B, 0xFF65}, {0xFFE0, 0xFFE6}, {0xFFE8, 0xFFEE}, {0xFFFC, 0xFFFD},
  {0x10100, 0x10102}, {0x10137, 0x1013F}, {0x10179, 0x10189}, {0x1018C, 0x1018E}, {0x10190, 0x1019C},
  {0x101A0, 0x101A0}, {0x101D0, 0x101FC}, {0x1039F, 0x1039F}, {0x103D0, 0x103D0}, {0x1056F, 0x1056F},
  {0x10857, 0x10857}, {0x10877, 0x10878}, {0x1091F, 0x1091F}, {0x1093F, 0x1093F}, {0x10A50, 0x10A58},
  {0x10A7F, 0x10A7F}, {0x10AC8, 0x10AC8}, {0x10AF0, 0x10AF6}, {0x10B39, 0x10B3F}, {0x10B99, 0x10B9C},
  {0x10EAD, 0x10EAD}, {0x10F55, 0x10F59}, {0x10F86, 0x10F89}, {0x11047, 0x1104D}, {0x110BB, 0x110BC},
  {0x110BE, 0x110C1}, {0x11140, 0x11143}, {0x11174, 0x11175}, {0x111C5, 0x111C8}, {0x111CD, 0x111CD},
  {0x111DB, 0x111DB}, {0x111DD, 0x111DF}, {0x11238, 0x1123D}, {0x112A9, 0x112A9}, {0x1144B, 0x1144F},
  {0x1145A, 0x1145B}, {0x1145D, 0x1145D}, {0x114C6, 0x114C6}, {0x115C1, 0x115D7}, {0x11641, 0x11643},
  {0x11660, 0x1166C}, {0x116B9, 0x116B9}, {0x1173C, 0x1173F}, {0x1183B, 0x1183B}, {0x11944, 0x11946},
  {0x119E2, 0x119E2}, {0x11A3F, 0x11A46}, {0x11A9A, 0x11A9C}, {0x11A9E, 0x11AA2}, {0x11B00, 0x11B09},
  {0x11C41, 0x11C45}, {0x11C70, 0x11C71}, {0x11EF7, 0x11EF8}, {0x11F43, 0x11F4F}, {0x11FD5, 0x11FF1},
  {0x11FFF, 0x11FFF}, {0x12470, 0x12474}, {0x12FF1, 0x12FF2}, {0x16A6E, 0x16A6F}, {0x16AF5, 0x16AF5},
  {0x16B37, 0x16B3F}, {0x16B44, 0x16B45}, {0x16E97, 0x16E9A}, {0x16FE2, 0x16FE2}, {0x1BC9C, 0x1BC9C},
  {0x1BC9F, 0x1BC9F}, {0x1CF50, 0x1CFC3}, {0x1D000, 0x1D0F5}, {0x1D100, 0x1D126}, {0x1D129, 0x1D164},
  {0x1D16A, 0x1D16C}, {0x1D183, 0x1D184}, {0x1D18C, 0x1D1A9}, {0x1D1AE, 0x1D1EA}, {0x1D200, 0x1D241},
  {0x1D245, 0x1D245}, {0x1D300, 0x1D356}, {0x1D6C1, 0x1D6C1}, {0x1D6DB, 0x1D6DB}, {0x1D6FB, 0x1D6FB},
  {0x1D715, 0x1D715}, {0x1D735, 0x1D735}, {0x1D74F, 0x1D74F}, {0x1D76F, 0x1D76F}, {0x1D789, 0x1D789},
  {0x1D7A9, 0x1D7A9}, {0x1D7C3, 0x1D7C3}, {0x1D800, 0x1D9FF}, {0x1DA37, 0x1DA3A}, {0x1DA6D, 0x1DA74},
  {0x1DA76, 0x1DA83}, {0x1DA85, 0x1DA8B}, {0x1E14F, 0x1E14F}, {0x1E2FF, 0x1E2FF}, {0x1E95E, 0x1E95F},
  {0x1ECAC, 0x1ECAC}, {0x1ECB0, 0x1ECB0}, {0x1ED2E, 0x1ED2E}, {0x1EEF0, 0x1EEF1}, {0x1F000, 0x1F02B},
  {0x1F030, 0x1F093}, {0x1F0A0, 0x1F0AE}, {0x1F0B1, 0x1F0BF}, {0x1F0C1, 0x1F0CF}, {0x1F0D1, 0x1F0F5},
  {0x1F10D, 0x1F1AD}, {0x1F1E6, 0x1F202}, {0x1F210, 0x1F23B}, {0x1F240, 0x1F248}, {0x1F250, 0x1F251},
  {0x1F260, 0x1F265}, {0x1F300, 0x1F6D7}, {0x1F6DC, 0x1F6EC}, {0x1F6F0, 0x1F6FC}, {0x1F700, 0x1F776},
  {0x1F77B, 0x1F7D9}, {0x1F7E0, 0x1F7EB}, {0x1F7F0, 0x1F7F0}, {0x1F800, 0x1F80B}, {0x1F810, 0x1F847},
  {0x1F850, 0x1F859}, {0x1F860, 0x1F887}, {0x1F890, 0x1F8AD}, {0x1F8B0, 0x1F8B1}, {0x1F900, 0x1FA53},
  {0x1FA60, 0x1FA6D}, {0x1FA70, 0x1FA7C}, {0x1FA80, 0x1FA88}, {0x1FA90, 0x1FABD}, {0x1FABF, 0x1FAC5},
  {0x1FACE, 0x1FADB}, {0x1FAE0, 0x1FAE8}, {0x1FAF0, 0x1FAF8}, {0x1FB00, 0x1FB92}, {0x1FB94, 0x1FBCA},
};

inline bool is_PS(uint32_t c) {  // P, S
  return c < 0x80 ? c > ' ' && c < 0x7F && !is_L(c) && !is_N(c) : in_table(kPunctSymbol, c);
}

// Flint's pre-tokenizer (Recursive's tokenizer.json): a Sequence of three Isolated splits, each run on every piece the
// one before made (its matches and the text between them are the pieces), then ByteLevel without its regex:
//   1. \p{N}{1,3}
//   2. [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+ (CJK ideographs, hiragana, katakana)
//   3. [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//      |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//      | ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// 3 is matched as pretokenize_tekken matches its regex. A piece is the whole text for the next split: \s+(?!\S) takes
// a piece's trailing whitespace whole, and text 3 does not match (a digit group, a ZWJ or a control character with no
// letter after it) is a piece of its own.
void pretokenize_flint(const std::string& s, std::vector<std::string>& out) {
  std::vector<CP> cps = to_cps(s);
  const size_t n = cps.size();
  auto at = [&](size_t i) { return cps[i].c; };
  auto emit = [&](size_t a, size_t b) {
    const uint32_t b0 = cps[a].off, b1 = b < n ? cps[b].off : uint32_t(s.size());
    out.emplace_back(s.substr(b0, b1 - b0));
  };
  // 3 at i, in a piece that ends at `end`: the end of the match, i if none
  auto match = [&](size_t i, size_t end) {
    auto P = [&](size_t k) { return k < end && !is_nl(at(k)) && !is_L(at(k)) && !is_N(at(k)); };
    auto A = [&](size_t k) { return k < end && (is_Lut(at(k)) || is_Lmo(at(k)) || is_M(at(k))); };
    auto B = [&](size_t k) { return k < end && (is_Ll(at(k)) || is_Lmo(at(k)) || is_M(at(k))); };
    auto Q = [&](size_t k) { return k < end && is_PS(at(k)); };
    // (?i:'s|'t|'re|'ve|'m|'ll|'d) at k: its end, else k (U+017F, the long s, folds to s)
    auto contraction = [&](size_t k) {
      if (k + 1 >= end || at(k) != '\'') return k;
      const uint32_t a = at(k + 1) == 0x17F ? 's' : at(k + 1) | 32, b = k + 2 < end ? at(k + 2) | 32 : 0;
      if (a == 's' || a == 't' || a == 'm' || a == 'd') return k + 2;
      if (((a == 'r' || a == 'v') && b == 'e') || (a == 'l' && b == 'l')) return k + 3;
      return k;
    };
    size_t j = i;
    // P? A* B+, then a contraction
    for (int pre = P(i) ? 1 : 0; pre >= 0 && j == i; --pre) {
      const size_t j0 = i + size_t(pre);
      size_t a_end = j0;
      while (A(a_end)) ++a_end;
      for (size_t k = a_end + 1; k-- > j0;) {
        if (!B(k)) continue;
        size_t e = k;
        while (B(e)) ++e;
        j = contraction(e);
        break;
      }
    }
    // P? A+ B*, then a contraction
    for (int pre = P(i) ? 1 : 0; pre >= 0 && j == i; --pre) {
      const size_t j0 = i + size_t(pre);
      size_t a_end = j0;
      while (A(a_end)) ++a_end;
      if (a_end == j0) continue;
      while (B(a_end)) ++a_end;
      j = contraction(a_end);
    }
    //  ?[\p{P}\p{S}]+[\r\n]*
    if (j == i) {
      size_t k = at(i) == ' ' && Q(i + 1) ? i + 1 : i;
      if (Q(k)) {
        while (Q(k)) ++k;
        while (k < end && is_nl(at(k))) ++k;
        j = k;
      }
    }
    // whitespace, as in the Qwen split, up to the piece's end
    if (j == i && is_space(at(i))) {
      size_t k = i;
      while (k < end && is_space(at(k))) ++k;
      size_t last_nl = end;
      for (size_t q = i; q < k; ++q)
        if (is_nl(at(q))) last_nl = q;
      if (last_nl != end) j = last_nl + 1;
      else if (k < end && k - i > 1) j = k - 1;
      else j = k;
    }
    return j;
  };
  // 3 on the piece [a, b): its matches and the text between them
  auto split3 = [&](size_t a, size_t b) {
    size_t gap = a;
    for (size_t i = a; i < b;) {
      const size_t j = match(i, b);
      if (j == i) {
        ++i;
        continue;
      }
      if (gap < i) emit(gap, i);
      emit(i, j);
      i = gap = j;
    }
    if (gap < b) emit(gap, b);
  };
  // 2 on the piece [a, b): runs of CJK ideographs and kana (U+4E00..9FA5, U+3040..30FF), and the text between them
  auto cjk = [&](size_t k) { return (at(k) >= 0x4E00 && at(k) <= 0x9FA5) || (at(k) >= 0x3040 && at(k) <= 0x30FF); };
  auto split2 = [&](size_t a, size_t b) {
    for (size_t i = a; i < b;) {
      size_t k = i + 1;
      while (k < b && cjk(k) == cjk(i)) ++k;
      split3(i, k);
      i = k;
    }
  };
  // 1: groups of up to three numbers, and the text between them
  for (size_t i = 0; i < n;) {
    size_t k = i + 1;
    if (is_N(at(i)))
      while (k < n && k - i < 3 && is_N(at(k))) ++k;
    else
      while (k < n && !is_N(at(k))) ++k;
    split2(i, k);
    i = k;
  }
}

std::string nfc(const std::string& s) {
  bool ascii = std::all_of(s.begin(), s.end(), [](char c) { return (unsigned char)c < 0x80; });
  if (ascii) return s;
  CFStringRef in = CFStringCreateWithBytes(nullptr, (const UInt8*)s.data(), CFIndex(s.size()), kCFStringEncodingUTF8, false);
  if (!in) return s;
  CFMutableStringRef m = CFStringCreateMutableCopy(nullptr, 0, in);
  CFStringNormalize(m, kCFStringNormalizationFormC);
  CFIndex len = CFStringGetLength(m), need = 0;
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false, nullptr, 0, &need);
  std::string out(size_t(need), '\0');
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false, (UInt8*)out.data(), need, nullptr);
  CFRelease(m);
  CFRelease(in);
  return out;
}

// GPT-2 byte <-> unicode mapping
void byte_unicode(uint32_t (&b2u)[256]) {
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    b2u[b] = keep ? uint32_t(b) : uint32_t(256 + n++);
  }
}

void put_utf8(std::string& s, uint32_t c) {
  if (c < 0x80) s += char(c);
  else if (c < 0x800) { s += char(0xC0 | (c >> 6)); s += char(0x80 | (c & 0x3F)); }
  else if (c < 0x10000) { s += char(0xE0 | (c >> 12)); s += char(0x80 | ((c >> 6) & 0x3F)); s += char(0x80 | (c & 0x3F)); }
  else { s += char(0xF0 | (c >> 18)); s += char(0x80 | ((c >> 12) & 0x3F)); s += char(0x80 | ((c >> 6) & 0x3F)); s += char(0x80 | (c & 0x3F)); }
}

} // namespace

const char* const Tokenizer::kFlintSplit[3] = {
    "\\p{N}{1,3}", "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    " ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"};

void Tokenizer::load(const std::string& path) {
  double t0 = now_ms();
  std::ifstream f(path);
  if (!f) fatal("cannot read %s", path.c_str());
  nlohmann::json j = nlohmann::json::parse(f);
  auto& model = j["model"];
  if (model["type"] != "BPE") fatal("only BPE tokenizers are supported");
  // Tekken (Mistral, Nemotron 3.5): its Split regex, no normalizer, whole words looked up first (ignore_merges).
  // Flint (Recursive): exactly its Sequence of three Isolated splits (pretokenize_flint), then ByteLevel without its
  // regex. Anything else gets Qwen's split with NFC.
  static const char* kTekken =
      "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
      "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
      "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  auto isolated = [](const nlohmann::json& p, const char* re) {  // Split on the regex `re`, Isolated, not inverted
    return p.is_object() && p.value("type", "") == "Split" && p.contains("pattern") && p["pattern"].is_object() &&
           p["pattern"].value("Regex", "") == re && p.value("behavior", "") == "Isolated" && !p.value("invert", false);
  };
  const auto& pt = j["pre_tokenizer"];
  if (pt.is_object() && pt.value("type", "") == "Sequence" && pt.contains("pretokenizers") &&
      pt["pretokenizers"].is_array()) {
    const auto& seq = pt["pretokenizers"];
    if (seq.size() == 4 && isolated(seq[0], kFlintSplit[0]) && isolated(seq[1], kFlintSplit[1]) &&
        isolated(seq[2], kFlintSplit[2]) && seq[3].is_object() && seq[3].value("type", "") == "ByteLevel" &&
        !seq[3].value("add_prefix_space", true) && !seq[3].value("use_regex", true))
      split_ = Split::Flint;
    else
      for (auto& p : seq)
        if (p.value("type", "") == "Split" && p.contains("pattern") && p["pattern"].contains("Regex") &&
            p["pattern"]["Regex"] == kTekken)
          split_ = Split::Tekken;
  }
  if (split_ != Split::Qwen) {
    nfc_ = !j["normalizer"].is_null();
    ignore_merges_ = model.value("ignore_merges", false);
  }
  uint32_t b2u[256];
  byte_unicode(b2u);
  std::unordered_map<uint32_t, uint8_t> u2b;
  for (int b = 0; b < 256; ++b) u2b[b2u[b]] = uint8_t(b);

  int max_id = 0;
  for (auto& [tok, id] : model["vocab"].items()) max_id = std::max(max_id, id.get<int>());
  for (auto& a : j["added_tokens"]) max_id = std::max(max_id, a["id"].get<int>());
  id_to_bytes_.assign(max_id + 1, "");
  special_.assign(max_id + 1, false);
  added_id_.assign(max_id + 1, false);
  for (auto& [tok, idj] : model["vocab"].items()) {
    int id = idj.get<int>();
    vocab_[tok] = id;
    std::string raw;
    for (auto& c : to_cps(tok)) {
      auto it = u2b.find(c.c);
      raw += it != u2b.end() ? char(it->second) : '?';
    }
    id_to_bytes_[id] = raw;
  }
  for (int b = 0; b < 256; ++b) {
    std::string u;
    put_utf8(u, b2u[b]);
    auto it = vocab_.find(u);
    byte_id_[b] = it == vocab_.end() ? -1 : it->second;
  }
  int rank = 0;
  for (auto& m : model["merges"]) {
    std::string a, b;
    if (m.is_array()) {
      a = m[0];
      b = m[1];
    } else {
      std::string s = m;
      size_t sp = s.find(' ');
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    }
    auto ia = vocab_.find(a), ib = vocab_.find(b), im = vocab_.find(a + b);
    if (ia != vocab_.end() && ib != vocab_.end() && im != vocab_.end())
      merges_[(uint64_t(uint32_t(ia->second)) << 32) | uint32_t(ib->second)] = {rank, im->second};
    ++rank;
  }
  for (auto& a : j["added_tokens"]) {
    int id = a["id"];
    std::string content = a["content"];
    added_.push_back({content, id});
    id_to_bytes_[id] = content;
    special_[id] = a.value("special", false);
    added_id_[id] = true;
  }
  std::sort(added_.begin(), added_.end(), [](auto& x, auto& y) { return x.first.size() > y.first.size(); });
  logi("tokenizer: %zu tokens, %zu merges, %zu added (%.0f ms)", vocab_.size(), merges_.size(), added_.size(),
       now_ms() - t0);
}

void Tokenizer::bpe(const std::string& word, std::vector<int>& out) const {
  {
    std::lock_guard<std::mutex> lk(cache_mu_);
    auto it = cache_.find(word);
    if (it != cache_.end()) {
      out.insert(out.end(), it->second.begin(), it->second.end());
      return;
    }
  }
  if (ignore_merges_) {
    std::string u;
    for (unsigned char b : word) {
      static const auto b2u = [] {
        std::array<uint32_t, 256> t{};
        uint32_t raw[256];
        byte_unicode(raw);
        for (int i = 0; i < 256; ++i) t[i] = raw[i];
        return t;
      }();
      put_utf8(u, b2u[b]);
    }
    auto it = vocab_.find(u);
    if (it != vocab_.end()) {
      out.push_back(it->second);
      return;
    }
  }
  std::vector<int> sym;
  sym.reserve(word.size());
  for (unsigned char b : word) sym.push_back(byte_id_[b]);
  while (sym.size() > 1) {
    int best_rank = INT32_MAX, best_i = -1, best_id = -1;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      auto it = merges_.find((uint64_t(uint32_t(sym[i])) << 32) | uint32_t(sym[i + 1]));
      if (it != merges_.end() && it->second.first < best_rank) {
        best_rank = it->second.first;
        best_i = int(i);
        best_id = it->second.second;
      }
    }
    if (best_i < 0) break;
    // merge every occurrence of this pair left to right
    std::vector<int> nxt;
    nxt.reserve(sym.size());
    int a = sym[best_i], b = sym[best_i + 1];
    for (size_t i = 0; i < sym.size(); ++i) {
      if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) {
        nxt.push_back(best_id);
        ++i;
      } else {
        nxt.push_back(sym[i]);
      }
    }
    sym.swap(nxt);
  }
  out.insert(out.end(), sym.begin(), sym.end());
  if (word.size() <= 64) {
    std::lock_guard<std::mutex> lk(cache_mu_);
    if (cache_.size() > 200000) cache_.clear();
    cache_.emplace(word, std::move(sym));
  }
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int>& out) const {
  if (text.empty()) return;
  std::vector<std::string> words;
  if (split_ == Split::Flint) pretokenize_flint(nfc_ ? nfc(text) : text, words);
  else if (split_ == Split::Tekken) pretokenize_tekken(nfc_ ? nfc(text) : text, words);
  else pretokenize(nfc_ ? nfc(text) : text, words);
  for (auto& w : words) bpe(w, out);
}

std::vector<int> Tokenizer::encode(const std::string& text, size_t split_at, int* split_index) const {
  std::vector<int> out;
  size_t start = 0, i = 0;
  while (i < text.size()) {
    bool hit = false;
    if (text[i] == '<' || text[i] == '[') {
      for (auto& [content, id] : added_) {
        if (text.compare(i, content.size(), content) == 0) {
          encode_plain(text.substr(start, i - start), out);
          if (split_index && i == split_at) *split_index = int(out.size());
          out.push_back(id);
          i += content.size();
          start = i;
          hit = true;
          break;
        }
      }
    }
    if (!hit) ++i;
  }
  encode_plain(text.substr(start), out);
  return out;
}

const std::string& Tokenizer::token_bytes(int id) const {
  static const std::string empty;
  return id >= 0 && id < int(id_to_bytes_.size()) ? id_to_bytes_[id] : empty;
}

bool Tokenizer::is_special(int id) const { return id >= 0 && id < int(special_.size()) && special_[id]; }
bool Tokenizer::is_added(int id) const { return id >= 0 && id < int(added_id_.size()) && added_id_[id]; }

int Tokenizer::find(const std::string& token) const {
  for (auto& [c, id] : added_)
    if (c == token) return id;
  auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string s;
  for (int id : ids) s += token_bytes(id);
  return s;
}

std::string StreamDecoder::push(int id) {
  pending_ += tok_.token_bytes(id);
  // emit the longest prefix that ends on a UTF-8 boundary
  size_t n = pending_.size(), cut = n;
  for (size_t back = 1; back <= 4 && back <= n; ++back) {
    unsigned char c = pending_[n - back];
    if ((c & 0xC0) == 0x80) continue;  // continuation byte
    int need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (int(back) < need) cut = n - back;  // incomplete sequence at the end
    break;
  }
  std::string out = pending_.substr(0, cut);
  pending_.erase(0, cut);
  return utf8_valid(out);
}

std::string StreamDecoder::flush() {
  std::string out;
  out.swap(pending_);
  return utf8_valid(out);
}

std::string utf8_valid(const std::string& s) {
  const size_t n = s.size();
  size_t i = 0;
  std::string out;
  bool fixed = false;
  while (i < n) {
    const unsigned char c = s[i];
    if (c < 0x80) {
      if (fixed) out += char(c);
      ++i;
      continue;
    }
    // continuation bytes after the lead byte, and the allowed range of the first one (no overlongs, surrogates
    // or code points past U+10FFFF)
    int need = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) need = 1;
    else if (c >= 0xE0 && c <= 0xEF) need = 2, lo = c == 0xE0 ? 0xA0 : 0x80, hi = c == 0xED ? 0x9F : 0xBF;
    else if (c >= 0xF0 && c <= 0xF4) need = 3, lo = c == 0xF0 ? 0x90 : 0x80, hi = c == 0xF4 ? 0x8F : 0xBF;
    size_t j = i + 1;
    int k = 0;
    for (; k < need && j < n; ++k, ++j) {
      const unsigned char d = s[j];
      if (d < (k ? 0x80 : lo) || d > (k ? 0xBF : hi)) break;
    }
    if (need > 0 && k == need) {
      if (fixed) out.append(s, i, j - i);
    } else {  // s[i, j) is the maximal subpart: the lead byte and the continuation bytes that fit it
      if (!fixed) out.assign(s, 0, i), fixed = true;
      out += "\xEF\xBF\xBD";
    }
    i = j;
  }
  return fixed ? out : s;
}

} // namespace ni
