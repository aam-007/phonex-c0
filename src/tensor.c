// Handles raw memory and matrix multiplication logic.

#include "../include/phonex.h"

Tensor tensor_alloc(int n, int d) {
    Tensor t;
    t.n = n; t.d = d;
    t.data = (float*)calloc(n * d, sizeof(float));
    t.grad = (float*)calloc(n * d, sizeof(float));
    return t;
}

void tensor_free(Tensor* t) {
    if(t->data) free(t->data);
    if(t->grad) free(t->grad);
}

void tensor_init_xavier(Tensor* t) {
    // [FIX] Reduce scale by 0.1 to prevent initial explosion
    float scale = sqrtf(6.0f / (float)(t->n + t->d)) * 0.1f;
    
    for (int i = 0; i < t->n * t->d; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        t->data[i] = (r * 2.0f - 1.0f) * scale;
    }
}

void tensor_zero_grad(Tensor* t) {
    memset(t->grad, 0, t->n * t->d * sizeof(float));
}

// C = A x B
void matmul(Tensor* A, Tensor* B, Tensor* C) {
    for (int i = 0; i < A->n; i++) {
        for (int j = 0; j < B->d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->d; k++) {
                sum += A->data[i * A->d + k] * B->data[k * B->d + j];
            }
            C->data[i * C->d + j] = sum;
        }
    }
}

// Backprop helper: C_grad += A^T * B_grad
void matmul_at_b_accum(Tensor* A, Tensor* B, Tensor* C_grad) {
    for (int i = 0; i < A->d; i++) {
        for (int j = 0; j < B->d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->n; k++) {
                sum += A->data[k * A->d + i] * B->grad[k * B->d + j];
            }
            C_grad->data[i * C_grad->d + j] += sum;
        }
    }
}

// Backprop helper: A_grad += C_grad * B^T
void matmul_c_bt_accum(Tensor* C, Tensor* B, Tensor* A_grad) {
    for (int i = 0; i < C->n; i++) {
        for (int j = 0; j < B->n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < C->d; k++) {
                sum += C->grad[i * C->d + k] * B->data[j * B->d + k];
            }
            A_grad->data[i * A_grad->d + j] += sum;
        }
    }
}