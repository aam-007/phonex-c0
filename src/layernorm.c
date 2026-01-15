#include "../include/phonex.h"

void forward_layernorm(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* out, Tensor* mean_cache, Tensor* var_cache) {
    for (int i = 0; i < x->n; i++) {
        float mean = 0.0f;
        float var = 0.0f;
        
        for (int j = 0; j < x->d; j++) mean += x->data[i * x->d + j];
        mean /= x->d;
        mean_cache->data[i] = mean;
        
        for (int j = 0; j < x->d; j++) {
            float diff = x->data[i * x->d + j] - mean;
            var += diff * diff;
        }
        var /= x->d;
        var_cache->data[i] = var;
        
        // [FIX] Ensure positive variance
        float inv_std = 1.0f / sqrtf(var + 1e-5f); 
        
        for (int j = 0; j < x->d; j++) {
            float norm = (x->data[i * x->d + j] - mean) * inv_std;
            out->data[i * x->d + j] = norm * gamma->data[j] + beta->data[j];
        }
    }
}

void backward_layernorm(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* y, Tensor* mean, Tensor* var) {
    int N = x->d;
    for (int i = 0; i < x->n; i++) {
        // [FIX] Stronger epsilon
        float inv_std = 1.0f / sqrtf(var->data[i] + 1e-5f);
        float dvar = 0.0f;
        float dmean = 0.0f;
        
        // Pointers for speed
        float* dx_row = &x->grad[i * N];
        float* dy_row = &y->grad[i * N];
        float* x_row = &x->data[i * N];
        float m_val = mean->data[i];
        
        for (int j = 0; j < N; j++) {
            float x_hat = (x_row[j] - m_val) * inv_std;
            
            // Accumulate parameter gradients
            gamma->grad[j] += dy_row[j] * x_hat;
            beta->grad[j] += dy_row[j];
            
            // Accumulate internal gradients
            float dxhat = dy_row[j] * gamma->data[j];
            dvar += dxhat * (x_row[j] - m_val);
            dmean += dxhat;
        }
        
        // Optimize backprop math to avoid powf(inv_std, 3) instability
        dvar *= -0.5f * (inv_std * inv_std * inv_std);
        dmean = (-dmean * inv_std) - (2.0f * dvar * 0.0f); // Simplification: Mean term often cancels
        
        // Re-calculate full dmean correctly
        dmean = 0.0f;
        for(int j=0; j<N; j++) {
             float dxhat = dy_row[j] * gamma->data[j];
             dmean += dxhat * -inv_std + dvar * 2.0f * (x_row[j] - m_val) / N;
        }

        for (int j = 0; j < N; j++) {
             float dxhat = dy_row[j] * gamma->data[j];
             dx_row[j] += dxhat * inv_std + dvar * 2.0f * (x_row[j] - m_val) / N + dmean / N;
        }
    }
}