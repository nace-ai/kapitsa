#include "gguf/gguf.h"
namespace ni::gguf {
bool type_block(uint32_t t,uint32_t* e,uint32_t* b){if(t!=kQ2_K)return false;*e=256;*b=84;return true;}
std::string type_name(uint32_t){return "Q2_K";}
}
