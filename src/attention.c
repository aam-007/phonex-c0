// Implements Scaled Dot-Product Attention.

#include "../include/phonex.h"

void forward_attention(Tensor* x, Transformer* m, Activations* c) {
    matmul(x, &m->w_q, &c->q);
    matmul(x, &m->w_k, &c->k);
    matmul(x, &m->w_v, &c->v);
    
    float scale = 1.0f / sqrtf((float)D_MODEL);
    
    // Q * K^T + Causal Mask
    for (int i = 0; i < c->att_scores.n; i++) {
        for (int j = 0; j < c->att_scores.d; j++) {
            if (j > i) {
                c->att_scores.data[i * c->att_scores.d + j] = -1e9f;
            } else {
                float score = 0.0f;
                for (int k = 0; k < D_MODEL; k++) {
                    score += c->q.data[i * D_MODEL + k] * c->k.data[j * D_MODEL + k];
                }
                c->att_scores.data[i * c->att_scores.d + j] = score * scale;
            }
        }
    }
    
    softmax(&c->att_scores, &c->att_probs);
    matmul(&c->att_probs, &c->v, &c->att_out);
    matmul(&c->att_out, &m->w_o, &c->o_out);
}

void backward_attention(Tensor* x, Transformer* m, Activations* c) {
    // 1. Output Projection
    backward_linear(&c->att_out, &m->w_o, NULL, &c->o_out);
    
    // 2. Weighted Sum
    matmul_c_bt_accum(&c->att_out, &c->v, &c->att_probs); // dProbs
    matmul_at_b_accum(&c->att_probs, &c->att_out, &c->v); // dV
    
    // 3. Softmax Backprop
    for (int i = 0; i < c->att_probs.n; i++) {
        float sum_grad_p = 0.0f;
        for (int j = 0; j < c->att_probs.d; j++) {
            sum_grad_p += c->att_probs.grad[i * c->att_probs.d + j] * c->att_probs.data[i * c->att_probs.d + j];
        }
        for (int j = 0; j < c->att_scores.d; j++) {
            float p = c->att_probs.data[i * c->att_probs.d + j];
            float dp = c->att_probs.grad[i * c->att_probs.d + j];
            c->att_scores.grad[i * c->att_scores.d + j] += p * (dp - sum_grad_p);
        }
    }
    
    // 4. Q and K
    float scale = 1.0f / sqrtf((float)D_MODEL);
    
    // dQ
    for (int i = 0; i < c->att_scores.n; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for (int k = 0; k <= i; k++) {
                sum += c->att_scores.grad[i * c->att_scores.d + k] * c->k.data[k * D_MODEL + j];
            }
            c->q.grad[i * D_MODEL + j] += sum * scale;
        }
    }
    
    // dK
    for (int i = 0; i < c->att_scores.d; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            float sum = 0.0f;
            for (int k = i; k < c->att_scores.n; k++) {
                sum += c->att_scores.grad[k * c->att_scores.d + i] * c->q.data[k * D_MODEL + j];
            }
            c->k.grad[i * D_MODEL + j] += sum * scale;
        }
    }

    backward_linear(x, &m->w_q, NULL, &c->q);
    backward_linear(x, &m->w_k, NULL, &c->k);
    backward_linear(x, &m->w_v, NULL, &c->v);
}