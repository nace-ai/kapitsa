#include "model/ngram.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sys/resource.h>
using json = nlohmann::json;
int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("cost-check MODEL CORPUS MODE");
    std::ifstream cf(std::string(argv[1])+"/config.json"), input(argv[2]);
    json c, corpus; cf >> c; input >> corpus;
    c=c.value("text_config",c);
    auto ej=c.value("eos_token_id",json(248044));
    int eos=ej.is_array()?ej[0].get<int>():ej.get<int>();
    auto hash=ni::NgramHash::make(c.at("vocab_size"),c.value("seed",1234),0,
        c.value("ngram_vocab_size_base",int64_t(20000000)),c.value("heads_per_ngram",8),c.value("ngram_size",3),eos);
    ni::NgramTable table;
    table.open({{std::string(argv[1])+"/ngram_table.bin","weight","scales","biases"}},ni::NgramTable::Fmt::Affine,c.at("hidden_size").get<int>()/hash.heads);
    const auto tokens=corpus.at("sequences")[0].get<std::vector<int>>();
    const bool diagnostic_off=std::string(argv[3])=="no-prefetch";
    auto clock=[] { return std::chrono::steady_clock::now(); };
    auto seconds=[&](auto start) { return std::chrono::duration<double>(clock()-start).count(); };
    auto cpu=[] { rusage r{};getrusage(RUSAGE_SELF,&r);return std::pair<double,double>{r.ru_utime.tv_sec+r.ru_utime.tv_usec*1e-6,r.ru_stime.tv_sec+r.ru_stime.tv_usec*1e-6}; };
    for(int T:{1,16,512,1536}) {
      if(tokens.size()<size_t(T)) throw std::runtime_error("short corpus");
      std::vector<int64_t> ids(size_t(T)*hash.heads);
      for(int t=0;t<T;++t) hash.ids(tokens.data(),t,ids.data()+size_t(t)*hash.heads);
      const size_t values=ids.size()*table.dim();
      std::vector<float> out(values+2,123456.25f);
      for(size_t r=0;r<ids.size();++r)table.row(ids[r],out.data()+1+r*table.dim());
      for(int phase:{0,1,2}) {
        if(diagnostic_off&&phase!=2)continue;
        auto run=[&] {
          if(phase==0)for(size_t r=0;r<ids.size();++r)table.row(ids[r],out.data()+1+r*table.dim());
          else if(phase==1)for(auto id:ids)table.prefetch(id);
          else table.gather(ids.data(),T,hash.heads,out.data()+1);
        };
        auto warm=clock();do {run();}while(seconds(warm)<0.25);
        int repeat=1;
        for(;;){auto start=clock();for(int i=0;i<repeat;++i)run();double t=seconds(start);if(t>=0.025||repeat>=131072){repeat=std::max(1,std::min(200000,int(repeat*0.10/std::max(t,1e-9))));break;}repeat*=2;}
        json samples=json::array();
        for(int sample=0;sample<9;++sample) {
          const auto c0=cpu();const auto start=clock();
          for(int i=0;i<repeat;++i)run();
          const double wall=seconds(start);const auto c1=cpu();
          samples.push_back({{"wall_us",wall*1e6/repeat},{"user_us",(c1.first-c0.first)*1e6/repeat},{"system_us",(c1.second-c0.second)*1e6/repeat},{"repeat",repeat}});
        }
        size_t nonzero=0;double sum=0;
        for(size_t i=1;i<=values;++i){if(!std::isfinite(out[i]))throw std::runtime_error("nonfinite");nonzero+=out[i]!=0;sum+=out[i];}
        if(!nonzero||out.front()!=123456.25f||out.back()!=123456.25f)throw std::runtime_error("guard or empty output");
        std::vector<double> us;for(auto& x:samples)us.push_back(x["wall_us"]);std::sort(us.begin(),us.end());
        std::cout<<json({{"tokens",T},{"heads",hash.heads},{"dim",table.dim()},{"mode",phase==0?"row":phase==1?"prefetch":"gather"},{"prefetch_disabled",diagnostic_off},{"samples",samples},{"median_us",us[4]},{"nonzero",nonzero},{"sum",sum},{"guards",true},{"finite",true}}).dump()<<std::endl;
      }
    }
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
