#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
namespace ni {
enum class NhFmt : uint8_t { F32, Q8, FP4, IQ2, BF16, FP8 };
void nh_row_bytes(NhFmt,int64_t,size_t&,size_t&);
size_t nh_src_row_bytes(NhFmt,int64_t);
void nh_repack_rows(NhFmt,int64_t,const uint8_t*,int64_t,uint8_t*,uint8_t*,bool);
void ref_nh_repack_rows(NhFmt,int64_t,const uint8_t*,int64_t,uint8_t*,uint8_t*,bool);
}
using F=ni::NhFmt;
using Fn=void(*)(F,int64_t,const uint8_t*,int64_t,uint8_t*,uint8_t*,bool);
static uint64_t cases=0;
static void require(bool good,const char*why) {if(!good){std::fprintf(stderr,"FAIL %s case%llu\n",why,(unsigned long long)cases);std::exit(1);}}
static void expect(const uint8_t*src,uint8_t*out,int64_t K,int64_t rows){
 for(int64_t r=0;r<rows;++r)
  for(int64_t group=0;group<K/16;++group)
   for(int j=0;j<16;++j){int d=4*((j%8)/2)+(j%2)+2*(j/8);out[r*K+group*16+d]=src[r*K+group*16+j];}
}
static void oracle_case(int64_t K,int64_t rows,int so,int qo,int seed,bool l2){
 ++cases;size_t bytes=size_t(K)*size_t(rows);size_t pre=64;
 std::vector<uint8_t>source(pre+so+bytes+64,0x6b),out(pre+qo+bytes+64,0xa5),ref=out,want=out;
 uint8_t*src=source.data()+pre+so;uint8_t*q=out.data()+pre+qo;uint8_t*qref=ref.data()+pre+qo;uint8_t*expected=want.data()+pre+qo;
 for(size_t i=0;i<bytes;++i)src[i]=uint8_t((i*131u+i/16u*17u+unsigned(seed)*29u)^((i/256u)*7u));
 auto unchanged=source;expect(src,expected,K,rows);
 ni::ref_nh_repack_rows(F::FP8,K,src,rows,qref,nullptr,l2);ni::nh_repack_rows(F::FP8,K,src,rows,q,nullptr,l2);
 require(out==want,"independent source-index oracle including canaries");require(out==ref,"reference/candidate byte equality");require(source==unchanged,"source unchanged");
}
static void complete_codes(){
 for(int j=0;j<16;++j)for(int code=0;code<256;++code){
  ++cases;uint8_t src[16],out[16],want[16];for(int i=0;i<16;++i)src[i]=uint8_t(i*19+23);src[j]=uint8_t(code);
  expect(src,want,16,1);ni::nh_repack_rows(F::FP8,16,src,1,out,nullptr,false);require(std::memcmp(out,want,16)==0,"all 256 raw codes at every lane");
 }
}
static void guards(){
 const size_t page=size_t(sysconf(_SC_PAGESIZE));
 for(int64_t K:{16,32,128,3136,4096})for(int64_t rows:{1,2,3}){
  ++cases;size_t bytes=size_t(K)*size_t(rows),mapped=(bytes+page-1)/page*page+page;
  auto*sm=static_cast<uint8_t*>(mmap(nullptr,mapped,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0));
  auto*qm=static_cast<uint8_t*>(mmap(nullptr,mapped,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0));require(sm!=MAP_FAILED&&qm!=MAP_FAILED,"guard mmap");
  require(mprotect(sm+mapped-page,page,PROT_NONE)==0&&mprotect(qm+mapped-page,page,PROT_NONE)==0,"guard protection");
  uint8_t*src=sm+mapped-page-bytes,*q=qm+mapped-page-bytes;for(size_t i=0;i<bytes;++i)src[i]=uint8_t(i*131u+29u);std::vector<uint8_t>want(bytes);expect(src,want.data(),K,rows);
  ni::nh_repack_rows(F::FP8,K,src,rows,q,nullptr,false);require(std::memcmp(q,want.data(),bytes)==0,"guard page source/output tail");munmap(sm,mapped);munmap(qm,mapped);
 }
}
static void other_formats(){
 for(F fmt:{F::F32,F::Q8,F::FP4,F::IQ2,F::BF16})for(int64_t K:{256,512,4096})for(int64_t rows:{0,1,3})for(bool l2:{false,true}){
  ++cases;size_t qb,sb;ni::nh_row_bytes(fmt,K,qb,sb);size_t rb=ni::nh_src_row_bytes(fmt,K);std::vector<uint8_t>src(rb*size_t(rows)+128),q(qb*size_t(rows)+128,0xa5),s(sb*size_t(rows)+128,0xb6),qr=q,sr=s;
  for(size_t i=0;i<src.size();++i)src[i]=uint8_t(i*7u+13u);auto copy=src;
  ni::ref_nh_repack_rows(fmt,K,src.data()+3,rows,qr.data()+5,sr.data()+7,l2);ni::nh_repack_rows(fmt,K,src.data()+3,rows,q.data()+5,s.data()+7,l2);require(q==qr&&s==sr,"unchanged-format bytes and canaries");require(src==copy,"unchanged-format source preserved");
 }
}
static void oracle(){
 for(int64_t K:{0,1,15,16,17,31,32,48,64,128,256,512,1536,3136,4096})for(int64_t rows:{0,1,3,8})for(int so:{0,1,7,15,16,31})for(int qo:{0,1,7,15,16,31})for(int seed:{0,1,7})for(bool l2:{false,true})oracle_case(K,rows,so,qo,seed,l2);
 complete_codes();guards();other_formats();std::printf("oracle PASS cases=%llu\n",(unsigned long long)cases);
}
static void bench(){
 std::puts("rows,K,bytes,pair,leg,arm,repeats,nanoseconds,gbps,output_gate");
 for(auto shape:std::vector<std::pair<int64_t,int64_t>>{{256,512},{1024,3136},{4096,4096},{8192,4096}}){
  auto[rows,K]=shape;size_t bytes=size_t(rows)*size_t(K);std::vector<uint8_t>src(bytes),q(bytes),want(bytes);for(size_t i=0;i<bytes;++i)src[i]=uint8_t(i*13u+i/16u*7u);expect(src.data(),want.data(),K,rows);
  int reps=std::max(1,int((256u*1024u*1024u)/bytes));
  for(Fn fn:{ni::ref_nh_repack_rows,ni::nh_repack_rows})for(int i=0;i<4;++i)fn(F::FP8,K,src.data(),rows,q.data(),nullptr,false);
  for(int pair=0;pair<8;++pair)for(int leg=0;leg<2;++leg){
   Fn fn=leg==0?ni::ref_nh_repack_rows:ni::nh_repack_rows;const char*arm=leg==0?"A":"B";auto start=std::chrono::steady_clock::now();
   for(int rep=0;rep<reps;++rep)fn(F::FP8,K,src.data(),rows,q.data(),nullptr,false);
   auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();require(q==want,"benchmark complete output gate");
   std::printf("%lld,%lld,%zu,%d,%d,%s,%d,%lld,%.8f,PASS\n",(long long)rows,(long long)K,bytes,pair,leg,arm,reps,(long long)ns,double(bytes)*reps/double(ns));std::fflush(stdout);
  }
 }
}
int main(int argc,char**argv){if(argc==2&&!std::strcmp(argv[1],"bench"))bench();else oracle();}
