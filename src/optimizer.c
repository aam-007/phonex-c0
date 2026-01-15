#include "../include/phonex.h"

void parameter_update(Tensor* t, float lr) {
    for (int i = 0; i < t->n * t->d; i++) {
        t->data[i] -= lr * t->grad[i];
        t->grad[i] = 0.0f; 
    }
}

void model_step(Transformer* m, float lr) {
    parameter_update(&m->token_emb, lr);
    parameter_update(&m->pos_emb, lr);
    parameter_update(&m->w_q, lr);
    parameter_update(&m->w_k, lr);
    parameter_update(&m->w_v, lr);
    parameter_update(&m->w_o, lr);
    parameter_update(&m->ln1_g, lr);
    parameter_update(&m->ln1_b, lr);
    parameter_update(&m->w1, lr);
    parameter_update(&m->b1, lr);
    parameter_update(&m->w2, lr);
    parameter_update(&m->b2, lr);
    parameter_update(&m->ln2_g, lr);
    parameter_update(&m->ln2_b, lr);
    parameter_update(&m->w_final, lr);
}