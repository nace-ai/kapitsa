// scratch (not committed): Neural Engine matmul throughput with runtime (IOSurface) weights, per operand type
#include "ane/ane.h"
#include "util/log.h"
#include <cstdio>
#include <random>
#include <string>
using namespace ni;
using namespace ni::ane;
static std::string dims(uint64_t r, uint64_t w) { return "[1, 1, " + std::to_string(r) + ", " + std::to_string(w) + "]"; }
static std::string tens(const char* ty, uint64_t r, uint64_t w) { return std::string("tensor<") + ty + ", " + dims(r, w) + ">"; }
int main(int argc, char** argv) {
  if (!unavailable().empty()) fatal("ANE unavailable: %s", unavailable().c_str());
  const uint32_t C = argc > 1 ? atoi(argv[1]) : 3584, K = argc > 2 ? atoi(argv[2]) : 2560, R = argc > 3 ? atoi(argv[3]) : 1024;
  Surface wi = make_surface(C, K, false), xi = make_surface(K, R, false), wf = make_surface(C, K, true),
          xf = make_surface(K, R, true), y = make_surface(C, R, true);
  std::mt19937 rng(1);
  for (size_t i = 0; i < wi.bytes; ++i) wi.ptr()[i] = uint8_t(rng() % 64);
  for (size_t i = 0; i < xi.bytes; ++i) xi.ptr()[i] = uint8_t(rng() % 64);
  for (size_t i = 0; i < wf.bytes / 2; ++i) ((_Float16*)wf.ptr())[i] = _Float16(float(int(rng() % 64) - 32) / 128.f);
  for (size_t i = 0; i < xf.bytes / 2; ++i) ((_Float16*)xf.ptr())[i] = _Float16(float(int(rng() % 64) - 32) / 128.f);
  const std::string unit = "fp16(0x1p-7)";
  auto out = [&](const std::string& m) {
    return "        " + y.buffer_type(C, R) + " y = tensor_to_tensor_buffer<ios17>(input = " + m +
           ", interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" + y.strides(y.rows) + "));\n";
  };
  auto in = [&](const Surface& s, const std::string& n, uint32_t r, uint32_t w) {
    return "        " + tens(s.mil_type().c_str(), r, w) + " " + n + "_t = tensor_buffer_to_tensor<ios17>(input = " + n + ");\n";
  };
  auto mm = [&](const std::string& a, const std::string& b) {
    return "        " + tens("fp16", C, R) + " m = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + a + ", y = " + b + ");\n";
  };
  struct V { std::string name; const Surface *w, *x; std::string body; };
  std::vector<V> vs;
  // A: int8 weights and inputs, both dequantized to fp16 (the split's form)
  vs.push_back({"mmA", &wi, &xi, in(wi, "w", C, K) + in(xi, "x", K, R) +
      "        " + tens("fp16", C, K) + " wd = dequantize(input = w_t, scale = " + unit + ");\n" +
      "        " + tens("fp16", K, R) + " xd = dequantize(input = x_t, scale = " + unit + ");\n" + mm("wd", "xd") + out("m")});
  // B: fp16 weights and inputs
  vs.push_back({"mmB", &wf, &xf, in(wf, "w", C, K) + in(xf, "x", K, R) + mm("w_t", "x_t") + out("m")});
  // C: fp16 weights, int8 inputs dequantized
  vs.push_back({"mmC", &wf, &xi, in(wf, "w", C, K) + in(xi, "x", K, R) +
      "        " + tens("fp16", K, R) + " xd = dequantize(input = x_t, scale = " + unit + ");\n" + mm("w_t", "xd") + out("m")});
  // D: int8 straight into matmul (may not compile)
  vs.push_back({"mmD", &wi, &xi, in(wi, "w", C, K) + in(xi, "x", K, R) + mm("w_t", "x_t") + out("m")});
  std::vector<_Float16> one(1, _Float16(0));
  for (auto& v : vs) {
    std::string mil = "program(1.3)\n{\n    func " + v.name + "<ios18>(" + v.w->buffer_type(C, K) + " w, " + v.x->buffer_type(K, R) +
                      " x) {\n" + v.body + "    } -> (y);\n}\n";
    try {
      Program p(mil, fp16_blob(one.data(), 1), "zz_ane_mm_" + v.name + "_" + std::to_string(C) + "_" + std::to_string(K) + "_" + std::to_string(R));
      auto b = p.bind(p.proc(v.name), {v.w, v.x}, y);
      double best = 1e9;
      for (int i = 0; i < 12; ++i) best = std::min(best, p.evaluate(*b));
      const double fl = 2.0 * C * K * R;
      printf("%s: compile %.0f ms, best eval %.3f ms, %.2f TFLOP/s\n", v.name.c_str(), p.compile_ms, best, fl / (best * 1e-3) / 1e12);
    } catch (const std::exception& e) {
      printf("%s: failed: %.300s\n", v.name.c_str(), e.what());
    }
  }
}
