#include "model/ngram.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json = nlohmann::json;
int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("gather-check MODEL CORPUS OUTPUT");
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
    std::ofstream raw(argv[3],std::ios::binary);
    for(int T:{1,16,512,1536}) {
      if(tokens.size()<size_t(T)) throw std::runtime_error("short corpus");
      std::vector<int64_t> ids(size_t(T)*hash.heads);
      for(int t=0;t<T;++t) hash.ids(tokens.data(),t,ids.data()+size_t(t)*hash.heads);
      const size_t values=ids.size()*table.dim();
      std::vector<float> out(values+2,123456.25f);
      const int repeat=T==1?200:T==16?40:T==512?2:1;
      for(bool gather:{false,true}) {
        auto run=[&] {
          if(gather) table.gather(ids.data(),T,hash.heads,out.data()+1);
          else for(size_t r=0;r<ids.size();++r) table.row(ids[r],out.data()+1+r*table.dim());
        };
        run();
        size_t nonzero=0;
        for(size_t i=1;i<=values;++i) {if(!std::isfinite(out[i]))throw std::runtime_error("nonfinite");nonzero+=out[i]!=0;}
        if(!nonzero || out.front()!=123456.25f || out.back()!=123456.25f)throw std::runtime_error("output guard or empty output");
        raw.write(reinterpret_cast<const char*>(out.data()+1),values*sizeof(float));
        std::vector<double> us;
        for(int sample=0;sample<12;++sample) {
          const auto start=std::chrono::steady_clock::now();
          for(int i=0;i<repeat;++i)run();
          us.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeat);
        }
        if(out.front()!=123456.25f || out.back()!=123456.25f)throw std::runtime_error("timing guard");
        auto last=std::vector<double>(us.begin()+2,us.end());std::sort(last.begin(),last.end());
        std::cout<<json({{"tokens",T},{"heads",hash.heads},{"dim",table.dim()},{"mode",gather?"gather":"row-loop"},{"repeat",repeat},{"us",us},{"median_last10_us",(last[4]+last[5])/2},{"nonzero_values",nonzero},{"guards",true},{"finite",true}}).dump()<<std::endl;
      }
    }
    if(!raw)throw std::runtime_error("write failed");
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
