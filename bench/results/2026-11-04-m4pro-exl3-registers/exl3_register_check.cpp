// Local differential experiment; REFERENCE.metal is the unmodified source from the recorded base commit.
#define main exl3_reference_tool_main
#include "exl3_check.cpp"
#undef main
#define main exl3_compile_tool_main
#include "ni_compile.cpp"
#undef main

#include <array>
#include <iostream>
#include <numeric>

namespace {
void replace_all(std::string& s, const std::string& from, const std::string& to) {
  size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); }
}

std::string reference_source(const char* path) {
  std::ifstream in(path);
  if (!in) fatal("cannot read reference %s", path);
  std::string s{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  if (s.find("threadgroup half Ws[16 * kExl3Wld]") == std::string::npos)
    fatal("reference must be the original shared-memory EXL3 kernel");
  replace_all(s, "#include \"common.h\"", "");
  replace_all(s, "exl3_", "exl3_ref_");
  replace_all(s, "kExl3", "kExl3Ref");
  replace_all(s, "Exl3Params", "Exl3RefParams");
  s += R"METAL(
[[kernel]] void exl3_fragment_probe(const device uint* w [[buffer(0)]], device half* out [[buffer(1)]],
                                  constant uint2& p [[buffer(2)]], uint tile [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  uint cw[8];
  exl3_lane_cw(w + tile * (p.x / 2), p.x, exl3_mma_lane(lane), cw);
  for (uint kr = 0; kr < 2; ++kr) for (uint nc = 0; nc < 2; ++nc) {
    const uint m = 2 * kr + 4 * nc;
    simdgroup_half8x8 f;
    f.thread_elements()[0] = half(exl3_mcg(cw[m] & p.y));
    f.thread_elements()[1] = half(exl3_mcg(cw[m + 1] & p.y));
    simdgroup_store(f, out + tile * 256 + kr * 128 + nc * 8, 16, ulong2(0, 0), true);
  }
}
)METAL";
  return s;
}

struct Guard {
  Buffer raw, b;
  explicit Guard(size_t size) {
    raw = Gpu::get().alloc(size + 512);
    b = raw.view(256, size);
    reset();
  }
  ~Guard() { Gpu::get().release(raw); }
  Guard(const Guard&) = delete;
  Guard& operator=(const Guard&) = delete;
  void reset() { memset(raw.ptr(), 0xa5, raw.size); }
  bool intact() const {
    for (size_t i = 0; i < 256; ++i)
      if (raw.as<uint8_t>()[i] != 0xa5 || raw.as<uint8_t>()[256 + b.size + i] != 0xa5) return false;
    return true;
  }
};

size_t differences(const Buffer& a, const Buffer& b) {
  if (a.size != b.size) fatal("comparison size mismatch");
  size_t bad = 0;
  for (size_t i = 0; i < a.size; ++i) bad += a.as<uint8_t>()[i] != b.as<uint8_t>()[i];
  return bad;
}

bool fragment_check(uint32_t n, uint32_t window, unsigned pattern, unsigned tiles) {
  Guard w(size_t(tiles) * n * 2), y(size_t(tiles) * 256 * 2);
  std::mt19937 rng(73 + n * 37 + pattern);
  for (size_t i = 0; i < w.b.size / 2; ++i)
    w.b.as<uint16_t>()[i] = pattern == 1 ? 0 : pattern == 2 ? 0xffff : uint16_t(rng());
  const uint32_t p[2]{n, (1u << window) - 1};
  auto e = Gpu::get().begin();
  e->use(Gpu::get().pipeline("exl3_fragment_probe"));
  e->buf(0, w.b); e->buf(1, y.b); e->bytes(2, p, sizeof(p));
  e->dispatch(tiles, 1, 1, 32); e->commit_and_wait();
  size_t bad = 0;
  for (unsigned t = 0; t < tiles; ++t) {
    float ref[256];
    exl3::decode_tile(w.b.as<uint16_t>() + size_t(t) * n, n, window, ref);
    for (unsigned i = 0; i < 256; ++i) bad += y.b.as<uint16_t>()[t * 256 + i] != f2h(ref[i]);
  }
  bool guards = w.intact() && y.intact();
  std::cout << json{{"kind","fragment"},{"n",n},{"window",window},{"pattern",pattern},{"tiles",tiles},
                    {"weight_bit_mismatches",bad},{"guards",guards}} << '\n';
  return !bad && guards;
}

struct Output {
  Guard hg, hu, md, yd;
  Output(uint32_t P, uint32_t D, uint32_t I) : hg(size_t(P)*I*4), hu(size_t(P)*I*4),
      md(size_t(P)*I*2), yd(size_t(P)*D*4) {}
};

struct PrefillCase {
  exl3::Shape sh;
  exl3::Experts X;
  uint32_t T, P, max_tiles;
  unsigned pattern_id;
  bool real_weights;
  Guard xh, idx, tiles, ntiles, xrows, yrows;
  Output out[2];
  Pipeline gu[2], dn[2], mid[2], fin[2];
  PrefillCase(uint32_t D, uint32_t I, uint32_t E, uint32_t k, uint32_t toks,
              uint32_t ng, uint32_t nu, uint32_t nd, uint32_t window, unsigned pattern, const char* fixture = nullptr)
      : sh{D,I,k,E,window}, T(toks), P(T*k), max_tiles((P+31)/32+E),
        pattern_id(pattern), real_weights(fixture != nullptr),
        xh(size_t(T)*D*2), idx(size_t(P)*4), tiles(size_t(max_tiles)*12), ntiles(16),
        xrows(size_t(P)*4), yrows(size_t(P)*4), out{Output(P,D,I),Output(P,D,I)} {
    std::mt19937 rng(519 + T + 31*ng + pattern);
    if (fixture) {
      WeightStore ws;
      const std::string prefix = "language_model.model.layers.0.mlp.switch_mlp.";
      ws.open_dir(fixture,[&](const std::string& key){return key.rfind(prefix,0)==0;},
                  [](const std::string& f){return f.find("-L00-")!=std::string::npos;});
      auto bank = [&](const char* proj, uint32_t in, uint32_t width, uint32_t n) {
        const Tensor& t = ws.get(prefix+proj+".trellis");
        const Tensor& su = ws.get(prefix+proj+".suh");
        const Tensor& sv = ws.get(prefix+proj+".svh");
        if (t.shape != std::vector<int64_t>{512,in/16,width/16,n} || E>512) fatal("unexpected fixture shape");
        return exl3::Proj{upload(t.buf.ptr(),size_t(E)*in/16*width/16*n*2),
                          upload(su.buf.ptr(),size_t(E)*in*2),upload(sv.buf.ptr(),size_t(E)*width*2),n};
      };
      X = {bank("gate_proj",D,I,ng),bank("up_proj",D,I,nu),bank("down_proj",I,D,nd)};
    } else {
      X = {device(random_bank(rng,E,D,I,ng)), device(random_bank(rng,E,D,I,nu)), device(random_bank(rng,E,I,D,nd))};
    }
    std::string why;
    if (!exl3::shapes_ok(sh,X,&why)) fatal("invalid case: %s", why.c_str());
    std::normal_distribution<float> normal(0.0f,0.3f);
    for (size_t i = 0; i < size_t(T)*D; ++i) xh.b.as<uint16_t>()[i] = f2h(pattern == 2 ? 0.0f : normal(rng));
    std::vector<std::vector<uint32_t>> by_expert(E);
    std::vector<uint32_t> ids(E); std::iota(ids.begin(),ids.end(),0);
    for (uint32_t t = 0; t < T; ++t) {
      std::shuffle(ids.begin(),ids.end(),rng);
      for (uint32_t j = 0; j < k; ++j) {
        const uint32_t expert = pattern == 1 ? j : ids[j], pair = t*k+j;
        idx.b.as<uint32_t>()[pair] = expert; by_expert[expert].push_back(pair);
      }
    }
    uint32_t row = 0, nt = 0;
    for (uint32_t expert = 0; expert < E; ++expert) {
      auto& pairs = by_expert[expert]; std::shuffle(pairs.begin(),pairs.end(),rng);
      for (size_t j = 0; j < pairs.size(); j += 32) {
        const uint32_t begin = row;
        for (size_t q = j; q < std::min(j+32,pairs.size()); ++q, ++row) {
          xrows.b.as<uint32_t>()[row] = pairs[q]/k; yrows.b.as<uint32_t>()[row] = pairs[q];
        }
        tiles.b.as<uint32_t>()[3*nt] = expert;
        tiles.b.as<uint32_t>()[3*nt+1] = begin; tiles.b.as<uint32_t>()[3*nt+2] = row; ++nt;
      }
    }
    if (row != P || nt > max_tiles) fatal("invalid sorted routing");
    ntiles.b.as<uint32_t>()[0] = nt;
    for (unsigned a = 0; a < 2; ++a) {
      const std::string pre = a ? "exl3_" : "exl3_ref_";
      gu[a]=Gpu::get().pipeline(pre+"gu_mm"); dn[a]=Gpu::get().pipeline(pre+"dn_mm");
      mid[a]=Gpu::get().pipeline(pre+"mid"); fin[a]=Gpu::get().pipeline(pre+"fin_pairs");
    }
  }
  void encode(Encoder& e, unsigned a, unsigned phase) {
    auto& o = out[a];
    if (phase == 0 && a == 1) {
      exl3::enc_prefill(e,X,sh,P,max_tiles,xh.b,tiles.b,ntiles.b,xrows.b,yrows.b,idx.b,o.hg.b,o.hu.b,o.md.b,o.yd.b);
      return;
    }
    const struct { uint32_t D,I,k,ng,nu,nd,mask,rows; } p{sh.D,sh.I,sh.k,X.g.n,X.u.n,X.d.n,(1u<<sh.window)-1,P};
    if (phase == 0 || phase == 1) {
      e.use(gu[a]); e.buf(0,X.g.t); e.buf(1,X.g.suh); e.buf(2,X.u.t); e.buf(3,X.u.suh);
      e.buf(4,xh.b); e.buf(5,tiles.b); e.buf(6,ntiles.b); e.buf(7,xrows.b); e.buf(8,o.hg.b); e.buf(9,o.hu.b);
      e.val(10,p); e.dispatch(sh.I/64,max_tiles,2,128);
    }
    if (phase == 0) {
      const uint32_t sorted = 1;
      e.use(mid[a]); e.buf(0,o.hg.b); e.buf(1,o.hu.b); e.buf(2,X.g.svh); e.buf(3,X.u.svh);
      e.buf(4,X.d.suh); e.buf(5,idx.b); e.buf(6,yrows.b); e.buf(7,o.md.b); e.val(8,p); e.val(9,sorted);
      e.dispatch(sh.I/128,(P+3)/4,1,128);
    }
    if (phase == 0 || phase == 2) {
      e.use(dn[a]); e.buf(0,X.d.t); e.buf(1,o.md.b); e.buf(2,tiles.b); e.buf(3,ntiles.b);
      e.buf(4,yrows.b); e.buf(5,o.yd.b); e.val(6,p); e.dispatch(sh.D/64,max_tiles,1,128);
    }
    if (phase == 0) {
      e.use(fin[a]); e.buf(0,o.yd.b); e.buf(1,X.d.svh); e.buf(2,idx.b); e.val(3,p);
      e.dispatch(sh.D/128,(P+3)/4,1,128);
    }
  }
  double run(unsigned a, unsigned phase = 0, unsigned repeat = 1) {
    auto e = Gpu::get().begin();
    for (unsigned i = 0; i < repeat; ++i) encode(*e,a,phase);
    e->commit_and_wait(); return e->gpu_ms()*1000/repeat;
  }
  json shape() const {
    return json{{"D",sh.D},{"I",sh.I},{"E",sh.E},{"k",sh.k},{"T",T},{"rates",{X.g.n,X.u.n,X.d.n}},
                {"window",sh.window},{"tiles",ntiles.b.as<uint32_t>()[0]},
                {"weights",real_weights?"checkpoint_layer0":"random"},
                {"routing",pattern_id==1?"concentrated":"uniform_random"},{"activation_pattern",pattern_id}};
  }
  bool check() {
    run(0); run(1);
    std::array<size_t,4> bad{differences(out[0].hg.b,out[1].hg.b),differences(out[0].hu.b,out[1].hu.b),
                           differences(out[0].md.b,out[1].md.b),differences(out[0].yd.b,out[1].yd.b)};
    bool guards = xh.intact() && idx.intact() && tiles.intact() && ntiles.intact() && xrows.intact() && yrows.intact();
    size_t nonfinite = 0;
    for (auto& o : out) {
      for (const Guard* v : {&o.hg,&o.hu,&o.md,&o.yd}) guards &= v->intact();
      for (const Guard* v : {&o.hg,&o.hu,&o.yd})
        for (size_t i = 0; i < v->b.size/4; ++i) nonfinite += !std::isfinite(v->b.as<float>()[i]);
      for (size_t i = 0; i < o.md.b.size/2; ++i) nonfinite += !std::isfinite(h2f(o.md.b.as<uint16_t>()[i]));
    }
    json j = shape(); j.update({{"kind","chain_check"},{"mismatch_bytes_hg_hu_md_yd",bad},{"guards",guards},{"nonfinite",nonfinite}});
    std::cout << j << '\n';
    return guards && !nonfinite && std::all_of(bad.begin(),bad.end(),[](size_t x){return x==0;});
  }
  void bench() {
    if (!check()) fatal("benchmark outputs differ");
    for (unsigned phase : {1u,2u,0u}) {
      run(0,phase,2); run(1,phase,2);
      const unsigned repeat = std::clamp(unsigned(std::ceil(6000/std::max(1.0,run(0,phase)))),1u,64u);
      std::vector<double> samples[2];
      for (unsigned arm : {0u,1u,0u,1u,0u,1u,0u,1u}) samples[arm].push_back(run(arm,phase,repeat));
      json j = shape(); j.update({{"kind","bench"},{"phase",phase==0?"chain":phase==1?"gate_up":"down"},
          {"repeat",repeat},{"reference_us",samples[0]},{"candidate_us",samples[1]}, {"thermal",Gpu::thermal_state()}});
      std::cout << j << '\n';
    }
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) fatal("usage: exl3_register_check REFERENCE.metal MODE [LAYER0_DIR]");
  const std::string mode = argv[2];
  if (mode!="small" && mode!="check" && mode!="bench" && mode!="resources" && mode!="real" && mode!="realbench")
    fatal("bad mode");
  const char* fixture = argc == 4 ? argv[3] : nullptr;
  if ((mode=="real" || mode=="realbench") != (fixture!=nullptr)) fatal("real modes require a layer-0 fixture");
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) fatal("GPU lock unavailable");
  Gpu::get().compile(reference_source(argv[1]));
  std::cout << json{{"kind","start"},{"head",KAPITSA_GIT},{"mode",mode},{"gpu",Gpu::get().name()},
                    {"cores",Gpu::get().core_count()},{"thermal",Gpu::thermal_state()}} << '\n';
  std::cout.setf(std::ios::unitbuf);
  if (mode == "resources") {
    const char* dir = getenv("NI_ARCHIVE_DIR");
    if (!dir) fatal("NI_ARCHIVE_DIR is required");
    for (const char* name : {"exl3_ref_gu_mm","exl3_gu_mm","exl3_ref_dn_mm","exl3_dn_mm"}) {
      const Pipeline p = Gpu::get().pipeline(name);
      std::string path = std::string(dir)+"/"+name+".bin";
      if (!Gpu::get().archive(name,path)) fatal("archive failed");
      std::ifstream in(path,std::ios::binary);
      Bytes b{std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
      long text,stack,gprs; kernel_stats(b,&text,&stack,&gprs);
      std::cout << json{{"kind","resources"},{"name",name},{"shared_bytes",p.tg_mem},{"max_threads",p.max_threads},
                        {"code_bytes",text},{"stack_bytes",stack},{"gprs",gprs}} << '\n';
    }
    return 0;
  }
  size_t count = 0, failed = 0;
  if (mode == "real") {
    for (uint32_t T : {8u,33u,128u,512u}) for (unsigned pattern : {0u,1u,2u}) {
      Gpu::AllocationScope scope;
      PrefillCase c(2560,640,16,10,T,32,32,32,15,pattern,fixture);
      ++count; failed += !c.check();
    }
  } else if (mode != "bench" && mode != "realbench") {
    for (uint32_t n=16; n<=64; n+=2)
      for (uint32_t window : {8u,12u,15u,16u})
        for (unsigned pattern=0; pattern<(mode=="small"?1u:3u); ++pattern) {
          ++count; failed += !fragment_check(n,window,pattern,mode=="small"?3:31);
        }
    if (failed) return 1;
    for (uint32_t n : {16u,24u,32u,40u,48u,64u})
      for (uint32_t T : mode=="small"?std::vector<uint32_t>{1,31,33}:std::vector<uint32_t>{7,32,37,97,300}) {
        Gpu::AllocationScope scope;
        PrefillCase c(512,128,8,4,T,n,n==32?40:n,n,15,T%3);
        ++count; failed += !c.check();
      }
    if (mode == "check") {
      for (uint32_t T : {8u,33u,128u,512u}) {
        Gpu::AllocationScope scope;
        PrefillCase c(2560,640,32,8,T,32,32,32,15,0);
        ++count; failed += !c.check();
      }
      for (uint32_t D : {1024u,4096u}) {
        Gpu::AllocationScope scope;
        PrefillCase c(D,384,8,4,37,18,34,62,16,1);
        ++count; failed += !c.check();
      }
    }
  } else {
    for (uint32_t T : {32u,128u,512u,2048u}) {
      Gpu::AllocationScope scope;
      PrefillCase c(2560,640,512,10,T,32,32,32,15,0,fixture);
      c.bench(); ++count;
    }
    for (uint32_t T : {128u,512u}) {
      Gpu::AllocationScope scope;
      PrefillCase c(2560,640,512,10,T,32,32,32,15,1,fixture);
      c.bench(); ++count;
    }
  }
  std::cout << json{{"kind","complete"},{"cases",count},{"failed",failed}} << '\n';
  return failed?1:0;
}
