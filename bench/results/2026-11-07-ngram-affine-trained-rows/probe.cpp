#include "model/ngram.h"
#include "tokenizer/tokenizer.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;
std::string read(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p);
  return {std::istreambuf_iterator<char>(f), {}};
}
int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("probe MODE MODEL INPUT OUTPUT");
    const std::string mode=argv[1], model=argv[2], input=argv[3], output=argv[4];
    const auto cfg=json::parse(read(model+"/config.json"));
    const auto tc=cfg.value("text_config",cfg);
    auto ej=tc.value("eos_token_id",json(248044));
    const int eos=ej.is_array()?ej[0].get<int>():ej.get<int>();
    auto hash=ni::NgramHash::make(tc.at("vocab_size"),tc.value("seed",1234),0,
        tc.value("ngram_vocab_size_base",int64_t(20000000)),tc.value("heads_per_ngram",8),tc.value("ngram_size",3),eos);
    if (mode=="rows") {
      auto rows=json::parse(read(input)).at("rows").get<std::vector<int64_t>>();
      ni::NgramTable table;
      table.open({{model+"/ngram_table.bin","weight","scales","biases"}},ni::NgramTable::Fmt::Affine,
                 tc.at("hidden_size").get<int>()/hash.heads);
      std::ofstream f(output,std::ios::binary);
      std::vector<float> v(table.dim()+2,123456.25f);
      size_t nonzero=0;
      for (auto r:rows) {
        table.row(r,v.data()+1);
        if (v.front()!=123456.25f || v.back()!=123456.25f) throw std::runtime_error("row guard changed");
        for(int i=1;i<=table.dim();++i) {
          if(!std::isfinite(v[i])) throw std::runtime_error("nonfinite row");
          nonzero+=v[i]!=0;
        }
        f.write(reinterpret_cast<const char*>(v.data()+1),table.dim()*sizeof(float));
      }
      if(!f || rows.empty() || !nonzero) throw std::runtime_error("empty row output");
      std::cout<<json({{"rows",rows.size()},{"dimension",table.dim()},{"nonzero_values",nonzero},{"finite",true},{"guards",true},{"mapped_bytes",table.file_bytes()}}).dump()<<"\n";
      return 0;
    }
    ni::Tokenizer tok;tok.load(model+"/tokenizer.json");
    std::vector<std::vector<int>> seqs;
    if(mode=="corpus") {
      auto text=read(input);std::vector<std::string> docs;
      for(size_t a=text.find("### FILE:");a!=std::string::npos;) {
        auto b=text.find("\n### FILE:",a+1);
        docs.push_back(text.substr(a,b==std::string::npos?std::string::npos:b+1-a));
        a=b==std::string::npos?b:b+1;
      }
      if(docs.empty())throw std::runtime_error("no corpus sections");
      for(int i=0;i<8;++i) {
        auto ids=tok.encode(docs[size_t(i)%docs.size()]);const size_t need=512+384*(7-i)+24+1;
        if(ids.size()<need)throw std::runtime_error("short corpus");
        ids.resize(need);seqs.push_back(std::move(ids));
      }
    } else if(mode=="text") seqs.push_back(tok.encode(read(input)));
    else if(mode=="ids" || mode=="decode") seqs=json::parse(read(input)).at("sequences").get<decltype(seqs)>();
    else throw std::runtime_error("unknown mode");
    std::set<int64_t> unique;
    std::vector<int64_t> ids(hash.heads);
    for(const auto& seq:seqs)for(size_t i=0;i<seq.size();++i){
      hash.ids(seq.data(),int(i),ids.data());unique.insert(ids.begin(),ids.end());
    }
    json result={{"sequences",seqs},{"rows",unique},{"heads",hash.heads},{"moduli",hash.mod},{"offsets",hash.off},{"eos",eos}};
    if(mode=="decode") {
      std::vector<std::string> texts;for(const auto& seq:seqs)texts.push_back(tok.decode(seq));result["texts"]=texts;
    }
    std::ofstream f(output);f<<result.dump()<<"\n";if(!f)throw std::runtime_error("output failure");
    std::cout<<json({{"sequences",seqs.size()},{"unique_rows",unique.size()},{"output",output}}).dump()<<"\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
