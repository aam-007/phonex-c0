#include "../include/phonex.h"

// Helper to clip gradients to avoid exploding values
float clip(float val, float limit) {
    if (val > limit) return limit;
    if (val < -limit) return -limit;
    return val;
}

void parameter_update(Tensor* t, float lr) {
    for (int i = 0; i < t->n * t->d; i++) {
        // [FIX] Gradient Clipping: Clamp gradients between -5 and 5
        // This is crucial for stability in raw C implementations
        float clipped_grad = clip(t->grad[i], 5.0f);
        
        t->data[i] -= lr * clipped_grad;
        
        // Zero grad after update
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