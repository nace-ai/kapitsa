#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
using namespace ni;
std::string read(const char* p) { std::ifstream f(p); return {std::istreambuf_iterator<char>(f), {}}; }
void replace(std::string& s, const std::string& a, const std::string& b) {
  for (size_t p=0;(p=s.find(a,p))!=std::string::npos;p+=b.size()) s.replace(p,a.size(),b);
}
std::string source(std::string s, const std::string& suffix, bool capture) {
  replace(s,"#include \"common.h\"", "#include <metal_stdlib>\nusing namespace metal;\n#define NI_INLINE static inline __attribute__((always_inline))");
  if (capture) replace(s,"tokens[p.out_index] = int(chosen);\n    out_probs[0] = topv[0];", "for (uint q=0;q<K;++q) { tokens[q]=int(topi[q]); out_probs[q]=topv[q]; }");
  replace(s,"topk_stage1", "topk_stage1"+suffix);
  replace(s,"sample_stage2", "sample_stage2"+suffix);
  return s;
}
struct Guard {
  Buffer all,b;
  size_t n;
  Guard(Gpu& g,size_t count):n(count) { all=g.alloc((n+32)*4);std::fill_n(all.as<uint32_t>(),n+32,0x5a5a5a5au);b=all.view(64,n*4); }
  bool okay() {for(size_t j=0;j<16;++j)if(all.as<uint32_t>()[j]!=0x5a5a5a5au||all.as<uint32_t>()[16+n+j]!=0x5a5a5a5au)return false;return true;}
};
int main(int argc,char**argv) {
  if(argc!=4)return 2;
  bench::GpuLock lock;if(!bench::GpuLock::held_by_caller()&&!lock.acquire())return 3;
  auto&g=Gpu::get();
  auto a=read(argv[1]),b=read(argv[2]);
  for(auto [name,s]:std::vector<std::pair<std::string,std::string>>{{"probe_a",source(a,"_probe_a",false)},{"probe_b",source(b,"_probe_b",false)},{"capture_a",source(a,"_capture_a",true)},{"capture_b",source(b,"_capture_b",true)}})g.compile_extra(name,s.c_str());
  Pipeline ps[2]={g.pipeline("sample_stage2_probe_a"),g.pipeline("sample_stage2_probe_b")};
  Pipeline ds[2]={g.pipeline("sample_stage2_probe_a_dstep"),g.pipeline("sample_stage2_probe_b_dstep")};
  Pipeline caps[2]={g.pipeline("sample_stage2_capture_a"),g.pipeline("sample_stage2_capture_b")};
  bool small=std::string(argv[3])=="small",timing=std::string(argv[3])=="bench";
  unsigned fails=0,cases=0;
  for(unsigned L:{1u,2u,3u,16u,32u,33u,63u,64u,65u,121u,122u,128u,129u}) {
    if(small&&L>2)continue;
    for(unsigned K:{1u,2u,3u,7u,8u,16u,20u,31u,32u,33u,64u}) {
      const unsigned N=L*K; if(N>4096)continue;
      Buffer cv=g.alloc(N*4),ci=g.alloc(N*4);Guard out[2]={Guard(g,64),Guard(g,64)},prob[2]={Guard(g,64),Guard(g,64)};
      Buffer depth=g.alloc(64*4); for(unsigned i=0;i<64;++i)depth.as<uint32_t>()[i]=i%3==0?0xffffffffu:i*3;
      unsigned output_index=0; bool dstep=false; unsigned rng_step=17;
      auto run=[&](int arm,bool cap,float temp,float top_p,unsigned seed,unsigned repeat){
        std::fill_n(out[arm].b.as<uint32_t>(),64,0x5a5a5a5au);std::fill_n(prob[arm].b.as<uint32_t>(),64,0x5a5a5a5au);
        auto e=g.begin();for(unsigned t=0;t<repeat;++t){e->use(cap?caps[arm]:(dstep?ds[arm]:ps[arm]));e->buf(0,cv);e->buf(1,ci);e->buf(2,out[arm].b);e->buf(3,prob[arm].b);e->val(4,kp::Sample{N,K,temp,top_p,seed,rng_step,output_index});if(!cap&&dstep)e->buf(5,depth);e->dispatch(1,1,1,1024);}e->commit_and_wait();return e->gpu_ms()*1000/repeat;
      };
      for(unsigned pat=0;pat<(timing?1u:5u);++pat){
        for(unsigned l=0;l<L;++l){std::vector<std::pair<float,unsigned>> list;
          for(unsigned k=0;k<K;++k){unsigned id=l*2048+k;float v=float(int((id*2246822519u^(id>>4))%100003)-50001)/1000;
            if(pat==1)v=float(id%3);if(pat==2)v=id%2?0.f:-0.f;if(pat==3)v=id%37==0?INFINITY:(id%13==0?-INFINITY:float(id%17));if(pat==4)v=-INFINITY;
            if(v>-INFINITY)list.push_back({v,id});}
          std::sort(list.begin(),list.end(),[](auto x,auto y){return x.first>y.first||(x.first==y.first&&x.second<y.second);});
          for(unsigned k=0;k<K;++k){cv.as<float>()[l*K+k]=k<list.size()?list[k].first:-INFINITY;ci.as<uint32_t>()[l*K+k]=k<list.size()?list[k].second:0xffffffffu;}
        }
        if(timing){for(int i=0;i<3;++i){run(0,false,.7f,.9f,42,256);run(1,false,.7f,.9f,42,256);}for(int pair=0;pair<6;++pair)for(int arm=0;arm<2;++arm)printf("{\"L\":%u,\"K\":%u,\"pair\":%d,\"arm\":\"%c\",\"us\":%.6f}\n",L,K,pair,arm?'B':'A',run(arm,false,.7f,.9f,42,256));}
        else {
          std::vector<unsigned> order(N);std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](unsigned x,unsigned y){float vx=cv.as<float>()[x],vy=cv.as<float>()[y];return vx>vy||(vx==vy&&ci.as<uint32_t>()[x]<ci.as<uint32_t>()[y]);});
          for(int arm=0;arm<2;++arm){run(arm,true,.7f,.9f,42,1);for(unsigned k=0;k<K;++k){unsigned idx=order[k];uint32_t expected_id=ci.as<uint32_t>()[idx];if(cv.as<float>()[idx]==-INFINITY)expected_id=*std::min_element(ci.as<uint32_t>(),ci.as<uint32_t>()+N);fails+=out[arm].b.as<uint32_t>()[k]!=expected_id||prob[arm].b.as<uint32_t>()[k]!=std::bit_cast<uint32_t>(cv.as<float>()[idx]);}}
          for(unsigned oi:{0u,7u,63u})for(bool dep:{false,true})for(float temp:{0.f,.7f,1.2f})for(float tp:{.1f,.9f,1.f})for(unsigned seed:{0u,42u}) {
            output_index=oi;dstep=dep;rng_step=17;
            run(0,false,temp,tp,seed,1);run(1,false,temp,tp,seed,1);
            for(unsigned i=0;i<64;++i) {
              fails+=out[0].b.as<uint32_t>()[i]!=out[1].b.as<uint32_t>()[i]||prob[0].b.as<uint32_t>()[i]!=prob[1].b.as<uint32_t>()[i];
              if(i!=oi)for(int arm=0;arm<2;++arm)fails+=out[arm].b.as<uint32_t>()[i]!=0x5a5a5a5au;
              if(i!=0)for(int arm=0;arm<2;++arm)fails+=prob[arm].b.as<uint32_t>()[i]!=0x5a5a5a5au;
            }
            auto token=out[1].b.as<uint32_t>()[oi],pv=prob[1].b.as<uint32_t>()[0];
            if(dep) {dstep=false;rng_step+=depth.as<uint32_t>()[oi];run(1,false,temp,tp,seed,1);fails+=token!=out[1].b.as<uint32_t>()[oi]||pv!=prob[1].b.as<uint32_t>()[0];}
            ++cases;
          }
          output_index=0;dstep=false;rng_step=17;
          for(int arm=0;arm<2;++arm)fails+=!out[arm].okay()||!prob[arm].okay();
          printf("L=%u K=%u pattern=%u cumulative_failures=%u\n",L,K,pat,fails);
        }
      }
      g.release(cv);g.release(ci);g.release(depth);for(int arm=0;arm<2;++arm){g.release(out[arm].all);g.release(prob[arm].all);}fflush(stdout);
    }
  }
  printf("cases=%u failures=%u\n",cases,fails);return fails?1:0;
}
