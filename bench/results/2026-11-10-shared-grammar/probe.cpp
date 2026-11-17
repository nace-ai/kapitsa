#include "engine/grammar.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
using namespace ni;
int main(int argc,char**argv){int levels=argc>1?atoi(argv[1]):8,fan=argc>2?atoi(argv[2]):4;
 ojson s={{"$ref","#/$defs/a0"},{"$defs",ojson::object()}};
 for(int i=0;i<levels;++i){ojson a=ojson::array();for(int j=0;j<fan;++j)a.push_back({{"$ref","#/$defs/a"+std::to_string(i+1)}});s["$defs"]["a"+std::to_string(i)]={{"anyOf",a}};}
 s["$defs"]["a"+std::to_string(levels)]={{"type","string"}};
 auto begin=std::chrono::steady_clock::now();auto g=json_grammar(s);GrammarMatcher m(g,nullptr);auto state=m.start();bool ok=m.advance_text(state,"\"x\"");double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();printf("levels=%d fan=%d bytes=%zu accepted=%d threads=%zu ms=%.6f\n",levels,fan,s.dump().size(),ok&&m.accepting(state),state.threads.size(),ms);
}
