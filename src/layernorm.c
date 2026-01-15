

#include "../include/phonex.h"

void forward_layernorm(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* out, Tensor* mean_cache, Tensor* var_cache) {
    for (int i = 0; i < x->n; i++) {
        float mean = 0.0f, var = 0.0f;
        for (int j = 0; j < x->d; j++) mean += x->data[i * x->d + j];
        mean /= x->d;
        mean_cache->data[i] = mean;
        
        for (int j = 0; j < x->d; j++) {
            float diff = x->data[i * x->d + j] - mean;
            var += diff * diff;
        }
        var /= x->d;
        var_cache->data[i] = var;
        
        float inv_std = 1.0f / sqrtf(var + EPSILON);
        for (int j = 0; j < x->d; j++) {
            float norm = (x->data[i * x->d + j] - mean) * inv_std;
            out->data[i * x->d + j] = norm * gamma->data[j] + beta->data[j];
        }
    }
}

void backward_layernorm(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* y, Tensor* mean, Tensor* var) {
    int N = x->d;
    for (int i = 0; i < x->n; i++) {
        float inv_std = 1.0f / sqrtf(var->data[i] + EPSILON);
        float dvar = 0.0f, dmean = 0.0f;
        
        for (int j = 0; j < N; j++) {
            float x_hat = (x->data[i*N + j] - mean->data[i]) * inv_std;
            gamma->grad[j] += y->grad[i*N + j] * x_hat;
            beta->grad[j] += y->grad[i*N + j];
            
            float dxhat = y->grad[i*N + j] * gamma->data[j];
            dvar += dxhat * (x->data[i*N + j] - mean->data[i]) * -0.5f * powf(inv_std, 3);
            dmean += dxhat * -inv_std;
        }
        
        for (int j = 0; j < N; j++) {
             float dxhat = y->grad[i*N + j] * gamma->data[j];
             x->grad[i*N + j] += dxhat * inv_std + dvar * 2.0f * (x->data[i*N + j] - mean->data[i]) / N + dmean / N;
        }
    }
}