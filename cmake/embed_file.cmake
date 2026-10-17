# Writes SRC (the chat page) into OUT as a C++ string literal: ni::<NAME>.
file(READ ${SRC} BODY)
file(WRITE ${OUT} "// generated from ${SRC}\nnamespace ni { extern const char* const ${NAME}; const char* const ${NAME} = R\"KAPITSAEMBED(${BODY})KAPITSAEMBED\"; }\n")
