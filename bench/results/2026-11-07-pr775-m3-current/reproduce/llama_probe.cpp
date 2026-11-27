#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
using Clock=std::chrono::steady_clock;
static std::vector<std::vector<llama_token>> sequences(const std::string& p) {
 std::ifstream in(p); if(!in) throw std::runtime_error("no sequences");
 std::vector<std::vector<llama_token>> out;std::string line;
 while(std::getline(in,line)){std::istringstream ss(line);std::vector<llama_token> row;int t;while(ss>>t)row.push_back(t);out.push_back(row);}return out;
}
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("model mode(dump|speed) input output");
 int lockfd=-1;if(!getenv("NI_LOCK_HELD")){lockfd=open("/tmp/kapitsa-gpu.lock",O_CREAT|O_RDWR,0644);if(lockfd<0||flock(lockfd,LOCK_EX))throw std::runtime_error("lock");}
 llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=999;
 auto*m=llama_model_load_from_file(argv[1],mp);if(!m)throw std::runtime_error("model load");
 auto cp=llama_context_default_params();cp.n_ctx=8192;cp.n_batch=2048;cp.n_ubatch=512;cp.n_threads=4;cp.n_threads_batch=4;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
 auto*c=llama_init_from_model(m,cp);if(!c)throw std::runtime_error("context");
 auto*v=llama_model_get_vocab(m);const int nv=llama_vocab_n_tokens(v);fprintf(stderr,"probe vocab=%d n_ctx=%u\n",nv,llama_n_ctx(c));
 auto feed=[&](llama_token*t,int n){for(int i=0;i<n;i+=2048){auto b=llama_batch_get_one(t+i,std::min(2048,n-i));if(llama_decode(c,b))throw std::runtime_error("decode");}};
 std::string mode=argv[2];
 if(mode=="dump"){
  std::string in=argv[3],out=argv[4];auto seq=sequences(in+"/sequences.txt");std::ifstream rec(in+"/records.txt");
  std::vector<std::vector<int>> lengths(seq.size());int sid,L;while(rec>>sid>>L)lengths.at(sid).push_back(L);
  std::ofstream logits(out+"/step.f32",std::ios::binary),prefill(out+"/prefill.f32",std::ios::binary);if(!logits||!prefill)throw std::runtime_error("output");
  for(size_t i=0;i<seq.size();++i){if(lengths[i].empty())continue;llama_memory_clear(llama_get_memory(c),true);int first=lengths[i][0]-1;feed(seq[i].data(),first);auto*values=llama_get_logits_ith(c,-1);prefill.write((char*)values,nv*4);
   int pos=first;for(int target:lengths[i]){if(target<=pos||target>(int)seq[i].size())throw std::runtime_error("record range");feed(seq[i].data()+pos,target-pos);pos=target;values=llama_get_logits_ith(c,-1);logits.write((char*)values,nv*4);}fprintf(stderr,"window %zu done\n",i);
  }
 }else if(mode=="speed"){
  auto seq=sequences(argv[3]);auto prompt=seq.at(0);std::ofstream out(argv[4]);
  std::ifstream textfile(std::filesystem::path(argv[3]).parent_path()/"speed-prompt.txt");std::string text((std::istreambuf_iterator<char>(textfile)),{});std::vector<llama_token> check(text.size()+100);
  int nt=llama_tokenize(v,text.data(),text.size(),check.data(),check.size(),false,true);if(nt<0)throw std::runtime_error("tokenize");check.resize(nt);if(check!=prompt)throw std::runtime_error("tokenizer ids differ");fprintf(stderr,"shared prompt tokenization matches (%d tokens)\n",nt);
  for(int rep=-1;rep<3;++rep){llama_memory_clear(llama_get_memory(c),true);auto start=Clock::now();feed(prompt.data(),rep<0?std::min<int>(64,prompt.size()):prompt.size());std::vector<int>ids;auto first=start,last=start;int n=rep<0?8:256;
   for(int i=0;i<n;++i){const float*l=llama_get_logits_ith(c,-1);llama_token t=std::max_element(l,l+nv)-l;ids.push_back(t);last=Clock::now();if(i==0)first=last;if(llama_vocab_is_eog(v,t))break;if(i+1<n)feed(&t,1);}
   if(rep>=0){double pf=std::chrono::duration<double>(first-start).count(),d=std::chrono::duration<double>(last-first).count();out<<"{\"rep\":"<<rep<<",\"prefill_s\":"<<pf<<",\"decode_s\":"<<d<<",\"tokens\":"<<ids.size()<<",\"tok_s\":"<<(ids.size()-1)/d<<",\"ids\":[";for(size_t j=0;j<ids.size();++j)out<<(j?",":"")<<ids[j];out<<"]}\n";out.flush();if(ids.size()!=256||llama_vocab_is_eog(v,ids.back()))throw std::runtime_error("short/EOS speed sample");}
  }
 }else throw std::runtime_error("mode");
 llama_free(c);llama_model_free(m);llama_backend_free();if(lockfd>=0)close(lockfd);return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}}
