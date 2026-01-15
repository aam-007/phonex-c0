#include "../include/phonex.h"

// ASCII Tokenizer
void tokenize_input(const char* text, int* out_ids, int len) {
    for (int i = 0; i < len; i++) {
        out_ids[i] = (int)text[i];
    }
}

char detokenize_id(int id) {
    return (char)id;
}