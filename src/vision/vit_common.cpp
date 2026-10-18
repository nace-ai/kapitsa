// What the image encoders' towers share on the GPU (see vit_common.h).
#include "vision/vit_common.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <dirent.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace ni::vision {

Pipeline vit_pipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

nlohmann::json vit_read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) return nlohmann::json::object();
  try {
    return nlohmann::json::parse(f);
  } catch (...) {
    return nlohmann::json::object();
  }
}

std::vector<std::string> vit_tensor_names(const std::string& dir) {
  std::vector<std::string> names;
  const nlohmann::json idx = vit_read_json(dir + "/model.safetensors.index.json");
  if (idx.contains("weight_map") && idx["weight_map"].is_object()) {
    for (auto& [k, v] : idx["weight_map"].items()) names.push_back(k);
    return names;
  }
  if (DIR* d = opendir(dir.c_str())) {  // single-file models: the headers
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() <= 12 || n.compare(n.size() - 12, 12, ".safetensors") != 0) continue;
      std::ifstream f(dir + "/" + n, std::ios::binary);
      uint64_t len = 0;
      if (!f.read(reinterpret_cast<char*>(&len), 8) || len > (256u << 20)) continue;
      std::string h(len, '\0');
      if (!f.read(h.data(), std::streamsize(len))) continue;
      try {
        for (auto& [k, v] : nlohmann::json::parse(h).items()) names.push_back(k);
      } catch (...) {
      }
    }
    closedir(d);
  }
  return names;
}

// ---- weights ----

VitWeights::~VitWeights() { release(); }

void VitWeights::open(const std::string& dir, const std::string& prefix) {
  prefix_ = prefix;
  // only the files that hold the tower's tensors (a sharded model's index says which)
  std::set<std::string> files;
  const nlohmann::json idx = vit_read_json(dir + "/model.safetensors.index.json");
  if (idx.contains("weight_map") && idx["weight_map"].is_object())
    for (auto& [k, v] : idx["weight_map"].items())
      if (k.rfind(prefix, 0) == 0 && v.is_string()) files.insert(v.get<std::string>());
  ws_ = std::make_unique<WeightStore>();
  ws_->open_dir(
      dir, [prefix](const std::string& n) { return n.rfind(prefix, 0) == 0; },
      files.empty() ? WeightStore::Filter() : [files](const std::string& f) { return files.count(f) != 0; },
      [](const std::string&) { return true; });
}

void VitWeights::close() { ws_.reset(); }

const Tensor* VitWeights::find(const std::string& name) const { return ws_ ? ws_->find(prefix_ + name) : nullptr; }

void VitWeights::read_bf16(const std::string& name, uint16_t* dst, size_t rows, size_t k, size_t ld) const {
  const Tensor* t = find(name);
  if (!t) fatal("vision: missing tensor %s%s", prefix_.c_str(), name.c_str());
  if (size_t(t->numel()) != rows * k)
    fatal("vision: %s%s has %lld values, expected %zu", prefix_.c_str(), name.c_str(), (long long)t->numel(), rows * k);
  std::vector<uint8_t> raw(size_t(t->numel()) * dtype_size(t->dtype));
  ws_->read_deferred(prefix_ + name, raw.data());
  for (size_t r = 0; r < rows; ++r) {
    uint16_t* d = dst + r * ld;
    for (size_t i = 0; i < k; ++i) {
      const size_t j = r * k + i;
      switch (t->dtype) {
        case DType::BF16: d[i] = reinterpret_cast<const uint16_t*>(raw.data())[j]; break;
        case DType::F16: d[i] = f32_to_bf16(float(reinterpret_cast<const _Float16*>(raw.data())[j])); break;
        case DType::F32: d[i] = f32_to_bf16(reinterpret_cast<const float*>(raw.data())[j]); break;
        default: fatal("vision: %s%s: unsupported dtype", prefix_.c_str(), name.c_str());
      }
    }
    std::fill(d + k, d + ld, uint16_t(0));
  }
}

Buffer VitWeights::alloc(size_t bytes, const char* label) {
  Buffer b = Gpu::get().alloc(bytes + 16, label, /*tracked=*/false);
  memset(b.ptr(), 0, b.size);
  owned_.push_back(b);
  return b;
}

VitLin VitWeights::lin(const std::string& name, uint32_t pad_n, uint32_t pad_k, bool bias) {
  const Tensor* t = find(name + ".weight");
  if (!t || t->shape.size() != 2) fatal("vision: missing or not 2-D: %s%s.weight", prefix_.c_str(), name.c_str());
  VitLin L;
  const uint32_t n = uint32_t(t->shape[0]), k = uint32_t(t->shape[1]);
  L.N = std::max(n, pad_n);
  L.K = std::max(k, pad_k);
  L.w = alloc(size_t(L.N) * L.K * 2, "vit_w");
  read_bf16(name + ".weight", L.w.as<uint16_t>(), n, k, L.K);
  L.b = alloc(size_t(L.N) * 2, "vit_b");
  if (bias) read_bf16(name + ".bias", L.b.as<uint16_t>(), 1, n, n);
  return L;
}

Buffer VitWeights::vec(const std::string& name) {
  const Tensor* t = find(name);
  if (!t) fatal("vision: missing tensor %s%s", prefix_.c_str(), name.c_str());
  Buffer b = alloc(size_t(t->numel()) * 2, "vit_v");
  read_bf16(name, b.as<uint16_t>(), 1, size_t(t->numel()), size_t(t->numel()));
  return b;
}

void VitWeights::make_resident() { Gpu::get().make_resident(owned_); }

void VitWeights::release() {
  Gpu& g = Gpu::get();
  for (Buffer& b : owned_) g.release(b);
  owned_.clear();
}

// ---- dispatches ----

void vit_gemm(Encoder& e, const VitLin& L, const Buffer& x, uint32_t x_ld, const Buffer& y, uint32_t y_ld, int M,
              bool add) {
  const char* n;
  uint32_t bm, bn, th;
  bool mfast = false;
  if (M >= 64 && L.N > 64) n = "qrf_f16_m32_n128_mf", bm = 32, bn = 128, th = 128, mfast = true;
  else if (M > 8) n = "qrf_f16_m16_n64", bm = 16, bn = 64, th = 128;
  else n = "qrf_f16_m8_n64", bm = 8, bn = 64, th = 64;
  const kp::Qmm p{uint32_t(M), L.N, L.K, x_ld, y_ld, add ? 1u : 0u, 64, 0, 0};
  e.use(vit_pipe(n));
  e.buf(0, x);
  e.buf(1, L.w);
  e.buf(2, L.w);
  e.buf(3, L.w);
  e.buf(4, y);
  e.val(5, p);
  const uint32_t mt = (uint32_t(M) + bm - 1) / bm, nt = (L.N + bn - 1) / bn;
  if (mfast) e.dispatch(mt, nt, 1, th);
  else e.dispatch(nt, mt, 1, th);
}

Encoder& VitCmds::at(double gflop) {
  if (cur && work > 0 && work + gflop > kVitCmdGflop) {
    cur->commit();
    done.push_back(std::move(cur));
  }
  if (!cur) {
    cur = Gpu::get().begin();
    work = 0;
  }
  work += gflop;
  return *cur;
}

void VitCmds::finish() {
  if (cur) {
    cur->commit();
    done.push_back(std::move(cur));
  }
  for (auto& e : done) e->wait();
  done.clear();
}

namespace {
struct VitLnParams {
  uint32_t rows, D, x_ld, y_ld;
  float eps;
  uint32_t has_pb;
};
struct VitActParams {
  uint32_t rows, N, ld, act, out_ld;
};
struct VitAttnParams {
  uint32_t N, ld, qb0;
  float scale;
};
struct VitQkvParams {
  uint32_t rows, r0, N, heads, HD;
};

// The attention kernel for head dimension HD: vit_attn_d<HD>_s4, or NI_VIT_ATTN's when it is one of that dimension
// (vit_attn_d<HD>_s{4,8}[r2]: simdgroups per threadgroup, blocks of 8 queries per simdgroup)
struct AttnKernel {
  std::string name;
  uint32_t sg = 4, rb = 1;
  uint32_t queries() const { return sg * rb * 8; }
};
AttnKernel attn_kernel(int HD) {
  AttnKernel a;
  a.name = "vit_attn_d" + std::to_string(HD) + "_s4";
  if (const char* v = getenv("NI_VIT_ATTN"); v && *v) {
    unsigned d = 0, sg = 0, rb = 1;
    if (sscanf(v, "vit_attn_d%u_s%ur%u", &d, &sg, &rb) < 2 || (sg != 4 && sg != 8) || (rb != 1 && rb != 2))
      fatal("NI_VIT_ATTN=%s: expected vit_attn_d<head dim>_s{4,8}[r2]", v);
    if (int(d) == HD) a.name = v, a.sg = sg, a.rb = rb;
  }
  return a;
}
}  // namespace

void vit_ln(Encoder& e, const Buffer& x, uint32_t x_ld, const Buffer* pending, const Buffer& w, const Buffer& b,
            const Buffer& out, uint32_t out_ld, int rows, int D, float eps) {
  const VitLnParams lp{uint32_t(rows), uint32_t(D), x_ld, out_ld, eps, pending ? 1u : 0u};
  e.use(vit_pipe("vit_ln"));
  e.buf(0, x);
  e.buf(1, pending ? *pending : w);
  e.buf(2, w);
  e.buf(3, b);
  e.buf(4, out);
  e.val(5, lp);
  e.dispatch((uint32_t(rows) + 7) / 8, 1, 1, 256);
}

void vit_ln_inplace(Encoder& e, const Buffer& x, uint32_t x_ld, const Buffer* pending, const Buffer& w, const Buffer& b,
                    int rows, int D, float eps) {
  const VitLnParams lp{uint32_t(rows), uint32_t(D), x_ld, 0, eps, pending ? 1u : 0u};
  e.use(vit_pipe("vit_ln_f32"));
  e.buf(0, x);
  e.buf(1, pending ? *pending : w);
  e.buf(2, w);
  e.buf(3, b);
  e.val(5, lp);
  e.dispatch((uint32_t(rows) + 7) / 8, 1, 1, 256);
}

void vit_act(Encoder& e, const char* kernel, const Buffer& y, uint32_t ld, const Buffer& bias, const Buffer& dst,
             uint32_t out_ld, int rows, int n, uint32_t act) {
  const VitActParams ap{uint32_t(rows), uint32_t(n), ld, act, out_ld};
  e.use(vit_pipe(kernel));
  e.buf(0, y);
  e.buf(1, bias);
  e.buf(2, dst);
  e.val(3, ap);
  e.dispatch((ld + 255) / 256, uint32_t(rows), 1, 256);
}

void vit_qkv_split(Encoder& e, const Buffer& y, const Buffer& bias, const Buffer& Q, const Buffer& K, const Buffer& V,
                   int rows, int r0, int N, int heads, int HD) {
  const VitQkvParams qp{uint32_t(rows), uint32_t(r0), uint32_t(N), uint32_t(heads), uint32_t(HD)};
  e.use(vit_pipe("vit_qkv_split"));
  e.buf(0, y);
  e.buf(1, bias);
  e.buf(2, Q);
  e.buf(3, K);
  e.buf(4, V);
  e.val(5, qp);
  e.dispatch((uint32_t(heads * HD) + 255) / 256, uint32_t(rows), 1, 256);
}

void vit_attention(VitCmds& cs, const Buffer& Q, const Buffer& K, const Buffer& V, const Buffer& out, int N, int ld,
                   int heads, int HD) {
  static std::mutex mu;
  static std::unordered_map<int, AttnKernel> kernels;
  AttnKernel ak;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = kernels.find(HD);
    if (it == kernels.end()) it = kernels.emplace(HD, attn_kernel(HD)).first;
    ak = it->second;
  }
  const double qb_gflop = 4.0 * ak.queries() * double(N) * HD * heads / 1e9;
  const uint32_t nqb = (uint32_t(N) + ak.queries() - 1) / ak.queries();
  const uint32_t qb_per = uint32_t(std::max(1.0, std::floor(kVitCmdGflop / qb_gflop)));
  for (uint32_t qb = 0; qb < nqb; qb += qb_per) {
    const uint32_t n = std::min(qb_per, nqb - qb);
    Encoder& e = cs.at(qb_gflop * n);
    const VitAttnParams ap{uint32_t(N), uint32_t(ld), qb, 1.0f / std::sqrt(float(HD))};
    e.use(vit_pipe(ak.name));
    e.buf(0, Q);
    e.buf(1, K);
    e.buf(2, V);
    e.buf(3, out);
    e.val(4, ap);
    e.dispatch(n, uint32_t(heads), 1, ak.sg * 32);
  }
}

}  // namespace ni::vision
