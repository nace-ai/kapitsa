#include "gguf/quant.h"
#include "gguf/gguf.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <vector>
namespace ni::gguf {
bool type_block(uint32_t t,uint32_t* e,uint32_t* b){if(t!=kMXFP4)return false;*e=32;*b=17;return true;}
std::string type_name(uint32_t){return "MXFP4";}
}
int main(){
constexpr int8_t levels[]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
size_t cases=0;
for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
 std::fesetround(mode);
 for(int scale=0;scale<256;++scale)for(int q=0;q<256;++q)for(int align:{0,1,7,15}){
 std::array<uint8_t,33> input{};input[align]=scale;std::memset(input.data()+align+1,q,16);
 std::array<float,40> out;out.fill(123);ni::gguf::dequantize(ni::gguf::kMXFP4,input.data()+align,out.data(),39);
 float d=std::ldexp(1.0f,scale-128);
 for(int j=0;j<32;++j){float expected=d*float(levels[j<16?(q&15):(q>>4)]);if(std::memcmp(&expected,&out[j],4)){printf("FAIL %d %d %d %d %a %a\n",mode,scale,q,j,expected,out[j]);return 1;}}
 for(int j=32;j<40;++j)if(out[j]!=123)return 2;
 ++cases;
 }}
std::fesetround(FE_TONEAREST);printf("PASS %zu exhaustive scale/packed-byte/alignment/rounding cases with tail guards\n",cases);
std::vector<uint8_t> in(17*32768);std::vector<float> out(32*32768);for(size_t i=0;i<in.size();++i)in[i]=uint8_t(i*17+123);
for(int pass=0;pass<8;++pass){auto start=std::chrono::steady_clock::now();for(int rep=0;rep<50;++rep)ni::gguf::dequantize(ni::gguf::kMXFP4,in.data(),out.data(),out.size());auto end=std::chrono::steady_clock::now();printf("ns_per_block %.4f\n",std::chrono::duration<double,std::nano>(end-start).count()/(50*32768));}
}
