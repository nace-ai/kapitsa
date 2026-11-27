#include "model/affine_coeff_fixed.h"
#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
int main(int argc,char**argv) {
  uint32_t N=argc>1?atoi(argv[1]):248320,G=argc>2?atoi(argv[2]):40,ks=G==40?2:8,rg=8;
  ni::AffinePackedCoeff im;
  im.blocks=size_t((N+15)/16)*((G+1)/2);
  im.words.resize(im.blocks*272); im.headers.resize(im.blocks);
  for(size_t i=0;i<im.words.size();++i)im.words[i]=uint32_t(i*2654435761u);
  for(size_t i=0;i<im.headers.size();++i)im.headers[i]=uint32_t(i);
  auto start=std::chrono::steady_clock::now();
  ni::interleave_packed_slices(im,N,G,ks,rg);
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  struct rusage ru{};getrusage(RUSAGE_SELF,&ru);
  CC_SHA256_CTX ctx;CC_SHA256_Init(&ctx);CC_SHA256_Update(&ctx,im.words.data(),CC_LONG(im.words.size()*4));CC_SHA256_Update(&ctx,im.headers.data(),CC_LONG(im.headers.size()*4));
  unsigned char h[32];CC_SHA256_Final(h,&ctx);
  printf("{\"N\":%u,\"groups\":%u,\"ks\":%u,\"rg\":%u,\"elapsed_ms\":%.6f,\"maxrss_bytes\":%ld,\"sha256\":\"",N,G,ks,rg,ms,ru.ru_maxrss);
  for(auto v:h)printf("%02x",v);puts("\"}");
}
