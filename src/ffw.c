// Feed-forward network logic including the shared backward_linear function.

#include "../include/phonex.h"

void backward_linear(Tensor* x, Tensor* w, Tensor* b, Tensor* y_out) {
    // dW = x^T * dy
    matmul_at_b_accum(x, y_out, w);
    
    // dx = dy * w^T
    for (int i = 0; i < y_out->n; i++) {
        for (int j = 0; j < w->n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < w->d; k++) {
                sum += y_out->grad[i * y_out->d + k] * w->data[j * w->d + k];
            }
            x->grad[i * x->d + j] += sum;
        }
    }
    
    // db
    if (b) {
        for (int i = 0; i < y_out->n; i++) {
            for (int j = 0; j < y_out->d; j++) {
                b->grad[j] += y_out->grad[i * y_out->d + j];
            }
        }
    }
}

void forward_ffn(Tensor* x, Transformer* m, Activations* c) {
    matmul(x, &m->w1, &c->ffn1);
    // ReLU
    for (int i = 0; i < c->ffn1.n; i++) {
        for (int j = 0; j < c->ffn1.d; j++) {
            float val = c->ffn1.data[i * c->ffn1.d + j] + m->b1.data[j];
            c->ffn_relu.data[i * c->ffn1.d + j] = val > 0 ? val : 0;
        }
    }
    matmul(&c->ffn_relu, &m->w2, &c->ffn2);
    // Bias 2
    for (int i = 0; i < c->ffn2.n; i++) {
        for (int j = 0; j < c->ffn2.d; j++) {
            c->ffn2.data[i * c->ffn2.d + j] += m->b2.data[j];
        }
    }
}

void backward_ffn(Tensor* x, Transformer* m, Activations* c) {
    backward_linear(&c->ffn_relu, &m->w2, &m->b2, &c->ffn2);
    // ReLU Backprop
    for (int i = 0; i < c->ffn_relu.n * c->ffn_relu.d; i++) {
        if (c->ffn1.data[i] + m->b1.data[i % m->b1.d] > 0) {
            c->ffn1.grad[i] += c->ffn_relu.grad[i];
        }
    }
    backward_linear(x, &m->w1, &m->b1, &c->ffn1);
}