#include "gguf/gguf.h"
namespace ni::gguf {bool type_block(uint32_t t,uint32_t* e,uint32_t* b){if(t!=kIQ2_XXS)return false;*e=256;*b=66;return true;}std::string type_name(uint32_t){return "IQ2_XXS";}}
