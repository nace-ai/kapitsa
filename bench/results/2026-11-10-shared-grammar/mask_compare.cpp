#include "engine/grammar.h"
#include "tokenizer/tokenizer.h"
#include <fstream>
#include <cstdio>
using namespace ni;
int main(int argc,char**argv){
 if(argc!=3)return 2;
 Tokenizer tok;tok.load(argv[1]);std::vector<int> eos;
 for(const char* t:{"<|im_end|>","<|endoftext|>"})if(tok.find(t)>=0)eos.push_back(tok.find(t));
 auto tt=std::make_shared<TokenTable>(tok,tok.vocab_size(),eos,tok.find("<tool_call>"),tok.find("</tool_call>"));
 std::ofstream out(argv[2],std::ios::binary);size_t rows=0;
 for(int levels:{0,1,3,5})for(int kind:{0,1,2}){
  ojson s={{"$ref","#/$defs/a0"},{"$defs",ojson::object()}};
  for(int i=0;i<levels;++i){ojson a=ojson::array();for(int j=0;j<4;++j)a.push_back({{"$ref","#/$defs/a"+std::to_string(i+1)}});s["$defs"]["a"+std::to_string(i)]={{"anyOf",a}};}
  s["$defs"]["a"+std::to_string(levels)]=kind==0?ojson::parse(R"({"anyOf":[{"const":"yes"},{"const":"no"}]})"):
   kind==1?ojson::parse(R"({"type":"object","properties":{"answer":{"enum":["yes","no"]},"n":{"type":"integer","minimum":0,"maximum":99}},"required":["answer"],"additionalProperties":false})"):
   ojson::parse(R"({"type":"array","items":{"anyOf":[{"type":"boolean"},{"type":"null"}]},"minItems":1,"maxItems":3})");
  GrammarMatcher m(json_grammar(s),tt);
  for(const std::string& text:{"","\"","\"y","\"yes\"","\"no\"","\"bad\"","{","{\"a","{\"answer\":","{\"answer\":\"yes\"}","{\"answer\":\"no\",\"n\":12}","[","[t","[true","[true,","[true,null]","[false,null,true]","[false,null,true,false]"}){
   auto state=m.start();bool ok=m.advance_text(state,text);std::vector<uint32_t> mask(tt->words(),0);
   if(ok)m.fill_mask(state,mask.data());
   uint32_t flags[2]={uint32_t(ok),uint32_t(ok&&m.accepting(state))};out.write(reinterpret_cast<const char*>(flags),sizeof(flags));out.write(reinterpret_cast<const char*>(mask.data()),mask.size()*sizeof(mask[0]));++rows;
  }
 }
 if(!out.good())return 3;
 printf("rows=%zu vocabulary=%d complete masks written\n",rows,tok.vocab_size());
}
