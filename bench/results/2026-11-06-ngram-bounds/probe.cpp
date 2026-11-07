#include "model/ngram.h"
#include "util/log.h"
#include <fstream>
#include <iostream>
#include <filesystem>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 ni::FatalRecovery recovery;
 std::string path=argv[1];
 std::string h=R"({"weight":{"dtype":"BF16","shape":[128,160],"data_offsets":[0,40960]}})";
 {std::ofstream f(path,std::ios::binary);uint64_t len=h.size();f.write((char*)&len,8);f.write(h.data(),h.size());}
 try{ni::NgramTable t;t.open({{path,"weight","",""}},ni::NgramTable::Fmt::BF16,160);std::cout<<"ADMITTED truncated table: rows="<<t.rows()<<" mapped="<<t.file_bytes()<<" declared_payload=40960; no row accessed\n";}
 catch(const ni::FatalError& e){std::cout<<"REJECTED "<<e.what()<<'\n';}
 std::filesystem::remove(path);
}
