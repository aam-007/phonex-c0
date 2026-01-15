#include "../include/phonex.h"

void softmax(Tensor* input, Tensor* output) {
    for (int i = 0; i < input->n; i++) {
        float max_val = -1e9f;
        // Subtract Max for numerical stability (prevents overflow)
        for (int j = 0; j < input->d; j++) {
            if (input->data[i * input->d + j] > max_val) 
                max_val = input->data[i * input->d + j];
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < input->d; j++) {
            // exp(x - max) is always <= 1.0, so it never overflows
            float exp_val = expf(input->data[i * input->d + j] - max_val);
            output->data[i * input->d + j] = exp_val;
            sum_exp += exp_val;
        }
        
        for (int j = 0; j < input->d; j++) {
            output->data[i * input->d + j] /= sum_exp;
        }
    }
}

float cross_entropy_loss(Activations* c, int* targets) {
    float loss = 0.0f;
    int len = c->logits.n;
    
    for (int i = 0; i < len; i++) {
        int target = targets[i];
        float prob = c->probs.data[i * VOCAB_SIZE + target];
        
        // [FIX 2] Clamp Probability: log(0) is -inf
        if (prob < 1e-7f) prob = 1e-7f;
        
        loss -= logf(prob);
        
        for (int j = 0; j < VOCAB_SIZE; j++) {
            float ind = (j == target) ? 1.0f : 0.0f;
            c->logits.grad[i * VOCAB_SIZE + j] = (c->probs.data[i * VOCAB_SIZE + j] - ind) / len;
        }
    }
    return loss / len;
}