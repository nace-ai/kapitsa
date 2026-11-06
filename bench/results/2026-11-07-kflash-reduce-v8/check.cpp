#include "metal/gpu.h"
#include "model/kparams.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>
using namespace ni;
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const int reps = atoi(argv[2]);
  std::ifstream in(argv[1]);
  const std::string extra((std::istreambuf_iterator<char>(in)), {});
  if (extra.empty()) return 2;
  Gpu& g = Gpu::get();
  g.compile(extra);
  const Pipeline original = g.pipeline("dfl_attn_reduce_original"), normal = g.pipeline("dfl_attn_reduce");
  const Pipeline packed = g.pipeline("dfl_attn_reduce_v8"), perm = g.pipeline("dfl_x8_to_v8");
  printf("device=%s thermal=%s tg_original=%u tg_normal=%u tg_packed=%u reps=%d\n", g.name().c_str(),
         Gpu::thermal_state(), original.tg_mem, normal.tg_mem, packed.tg_mem, reps);
  Buffer po = g.alloc(size_t(64)*32*16*128*4), ml = g.alloc(size_t(64)*32*16*2*4), gst = g.alloc(64);
  Buffer oldx = g.alloc(2*16*4096*2+512), newx = g.alloc(oldx.size), ov = g.alloc(16*4096*2+512);
  Buffer nv = g.alloc(ov.size), os = g.alloc(2*512*4+512), ns = g.alloc(os.size), nx = g.alloc(os.size);
  std::mt19937 rng(702);
  std::normal_distribution<float> nd;
  int cases=0, fails=0;
  for (int nr : {8,16}) for (uint32_t split : {1u,2u,4u,8u,16u,32u,64u}) for (int plan=0;plan<3;++plan) {
    kp::DflAttnRed p{split, 700, 0, 32, uint32_t(nr)};
    uint32_t active=split;
    if (plan) {
      p.P=0xffffffffu; p.chunk=plan==2 ? 0x80000080u : 0;
      gst.as<uint32_t>()[0]=plan==1 ? 85 : 9000;
      const uint32_t P=gst.as<uint32_t>()[0], lo=P>2047 ? P-2047 : 0;
      const uint32_t chunk=std::max(plan==2 ? 128u : 32u, ((P-lo+split-1)/split+31)/32*32);
      active=std::max(1u,(P-lo+chunk-1)/chunk);
    }
    const size_t count=size_t(64)*32*nr;
    std::fill(po.as<float>(),po.as<float>()+count*128,NAN);
    std::fill(ml.as<float>(),ml.as<float>()+count*2,NAN);
    for (uint32_t s=0;s<active;++s) for (int h=0;h<32;++h) for (int r=0;r<nr;++r) {
      const size_t b=(size_t(s)*32+h)*nr+r;
      const bool empty=s>0 && (s+h+r)%5==0;
      ml.as<float>()[2*b]=empty ? -INFINITY : nd(rng)*20;
      ml.as<float>()[2*b+1]=empty ? NAN : 1+std::abs(nd(rng))*10;
      for (int d=0;d<128;++d) po.as<float>()[b*128+d]=empty ? NAN : nd(rng)*10;
    }
    for (Buffer b : {oldx,newx,ov,nv,os,ns,nx}) memset(b.ptr(),0xa5,b.size);
    const size_t xb=size_t(nr)*2*4096*2,vb=size_t(nr)*4096*2,sb=size_t(nr/8)*512*4;
    auto red=[&](Encoder& e,Pipeline pipe,Buffer x,Buffer sums,size_t bytes) {
      e.use(pipe); e.buf(0,po); e.buf(1,ml); e.buf(2,x.view(256,bytes)); e.buf(3,sums.view(256,sb));
      e.val(4,p); e.buf(5,gst); e.dispatch(32,nr,1,128);
    };
    auto base=[&](Encoder& e) {
      red(e,original,oldx,os,xb);
      for (int t=0;t<nr/8;++t) {
        e.use(perm); e.buf(0,oldx.view(256+size_t(t)*16*4096*2,16*4096*2));
        e.buf(1,ov.view(256+size_t(t)*8*4096*2,8*4096*2)); e.val(2,kp::DflPerm{4096}); e.dispatch(8,1,1,256);
      }
    };
    auto candidate=[&](Encoder& e) { red(e,packed,nv,ns,vb); };
    {auto e=g.begin(); base(*e); candidate(*e); red(*e,normal,newx,nx,xb); e->commit_and_wait();}
    bool ok=!memcmp(oldx.ptr(),newx.ptr(),oldx.size) && !memcmp(ov.ptr(),nv.ptr(),ov.size)
            && !memcmp(os.ptr(),ns.ptr(),os.size) && !memcmp(os.ptr(),nx.ptr(),os.size);
    for (Buffer b : {oldx,newx,ov,nv,os,ns,nx}) {
      const size_t used=b.mtl==oldx.mtl || b.mtl==newx.mtl ? xb : b.mtl==os.mtl || b.mtl==ns.mtl || b.mtl==nx.mtl ? sb : vb;
      for (size_t i=0;i<b.size;++i) if (i<256 || i>=256+used) ok &= b.as<uint8_t>()[i]==0xa5;
    }
    for (size_t i=0;i<vb/2;++i) ok &= std::isfinite(float(nv.view(256,vb).as<__fp16>()[i]));
    printf("case nr=%d splits=%u plan=%d active=%u bytes=%zu %s\n",nr,split,plan,active,vb,ok ? "PASS" : "FAIL");
    ++cases; fails+=!ok;
    if (reps>0 && nr==8 && plan==0 && (split==8 || split==16 || split==32)) {
      for (int arm=0;arm<6;++arm) {
        const bool b=arm%2;
        std::vector<double> times;
        for (int it=0;it<9;++it) {
          auto e=g.begin(); for (int rep=0;rep<reps;++rep) {if(b) candidate(*e);else base(*e);}
          e->commit_and_wait(); if(it>1) times.push_back(e->gpu_ms()*1000/reps);
        }
        printf("time splits=%u arm=%d variant=%c thermal=%s us=",split,arm,b?'B':'A',Gpu::thermal_state());
        for (double t:times) printf(" %.6f",t);
        puts("");
      }
    }
  }
  printf("cases=%d failures=%d\n",cases,fails);
  return fails ? 1 : 0;
}
