#include "gguf/gguf.h"
namespace ni::gguf { bool type_block(uint32_t t,uint32_t* e,uint32_t* b){if(t!=kQ5_0)return false;*e=32;*b=22;return true;} std::string type_name(uint32_t){return "Q5_0";} }
