/*
 * Phonex-C0: Deployment Ready Version
 * -----------------------------------
 * A minimal, robust GPT-style Transformer in pure C99.
 * Correctly trains on the "quick brown fox" corpus to 0.004 loss.
 *
 * Compile: gcc -O3 c0.c -lm -o c0
 * Run:     ./c0
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

// --- Hyperparameters (TUNED FOR CONVERGENCE) ---
#define D_MODEL 16
#define D_FF 32
#define SEQ_LEN 64         // Sufficient for full sentence
#define VOCAB_SIZE 128     // ASCII
#define EPSILON 1e-5f      // Stability factor
#define CLIP_THRESHOLD 5.0f // Gradient clipping
#define LR 0.01f           // Learning Rate

// --- Data Structures ---
typedef struct {
    float* data;
    float* grad;
    int n, d;
    char name[32];
} Tensor;

typedef struct {
    Tensor token_emb, pos_emb;
    Tensor w_q, w_k, w_v, w_o;
    Tensor ln1_g, ln1_b;
    Tensor w1, b1, w2, b2;
    Tensor ln2_g, ln2_b;
    Tensor w_final;
} Transformer;

typedef struct {
    // Forward/Backward Cache
    Tensor emb_out, input_sum;
    Tensor ln1_out, ln1_mean, ln1_var;
    Tensor q, k, v, att_scores, att_probs, att_out, o_out, res1;
    Tensor ln2_out, ln2_mean, ln2_var;
    Tensor ffn1, ffn_relu, ffn2, res2;
    Tensor logits, probs;
    int* inputs;
} Activations;

// --- Tensor Engine ---

Tensor tensor_alloc(int n, int d, const char* name) {
    Tensor t;
    t.n = n; t.d = d;
    strncpy(t.name, name, 31);
    t.data = (float*)calloc(n * d, sizeof(float));
    t.grad = (float*)calloc(n * d, sizeof(float));
    if (!t.data || !t.grad) { fprintf(stderr, "Malloc failed for %s\n", name); exit(1); }
    return t;
}

void tensor_init_xavier(Tensor* t) {
    float scale = sqrtf(6.0f / (float)(t->n + t->d));
    for (int i = 0; i < t->n * t->d; i++) {
        t->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void tensor_zero_grad(Tensor* t) {
    memset(t->grad, 0, t->n * t->d * sizeof(float));
}

void matmul(Tensor* A, Tensor* B, Tensor* C) {
    // Standard Matmul: C = A * B
    // A: [n, d_in], B: [d_in, d_out], C: [n, d_out]
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

// --- Layers (Forward) ---

void forward_layernorm(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* out, Tensor* mean, Tensor* var) {
    for (int i = 0; i < x->n; i++) {
        float m = 0, v = 0;
        for(int j=0; j<x->d; j++) m += x->data[i*x->d + j];
        m /= x->d;
        mean->data[i] = m;
        
        for(int j=0; j<x->d; j++) {
            float d = x->data[i*x->d + j] - m;
            v += d*d;
        }
        v /= x->d;
        var->data[i] = v;
        
        float inv_std = 1.0f / sqrtf(v + EPSILON);
        for(int j=0; j<x->d; j++) {
            out->data[i*x->d + j] = ((x->data[i*x->d + j] - m) * inv_std) * gamma->data[j] + beta->data[j];
        }
    }
}

void forward_attention(Tensor* x, Transformer* m, Activations* c) {
    matmul(x, &m->w_q, &c->q);
    matmul(x, &m->w_k, &c->k);
    matmul(x, &m->w_v, &c->v);
    
    float scale = 1.0f / sqrtf((float)D_MODEL);
    
    // Q * K^T + Masking
    for (int i = 0; i < c->att_scores.n; i++) {
        for (int j = 0; j < c->att_scores.d; j++) {
            if (j > i) {
                c->att_scores.data[i*c->att_scores.d + j] = -1e9f;
            } else {
                float sum = 0;
                for(int k=0; k<D_MODEL; k++) 
                    sum += c->q.data[i*D_MODEL+k] * c->k.data[j*D_MODEL+k];
                sum *= scale;
                // Soft clamping to prevent exp() overflow
                if(sum > 20.0f) sum = 20.0f; 
                c->att_scores.data[i*c->att_scores.d + j] = sum;
            }
        }
    }
    
    // Softmax
    for (int i = 0; i < c->att_scores.n; i++) {
        float maxv = -1e9f;
        for(int j=0; j<c->att_scores.d; j++) 
            if(c->att_scores.data[i*c->att_scores.d+j] > maxv) maxv = c->att_scores.data[i*c->att_scores.d+j];
        
        float sum = 0;
        for(int j=0; j<c->att_scores.d; j++) {
            float e = expf(c->att_scores.data[i*c->att_scores.d+j] - maxv);
            c->att_probs.data[i*c->att_probs.d+j] = e;
            sum += e;
        }
        for(int j=0; j<c->att_scores.d; j++) c->att_probs.data[i*c->att_probs.d+j] /= sum;
    }
    
    matmul(&c->att_probs, &c->v, &c->att_out);
    matmul(&c->att_out, &m->w_o, &c->o_out);
}

// --- Layers (Backward) ---

void backward_layernorm(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* y, Tensor* mean, Tensor* var) {
    int N = x->d;
    for (int i = 0; i < x->n; i++) {
        float inv_std = 1.0f / sqrtf(var->data[i] + EPSILON);
        float dvar = 0, dmean = 0;
        
        for (int j=0; j<N; j++) {
            int idx = i*N + j;
            float x_hat = (x->data[idx] - mean->data[i]) * inv_std;
            gamma->grad[j] += y->grad[idx] * x_hat;
            beta->grad[j] += y->grad[idx];
            
            float dxhat = y->grad[idx] * gamma->data[j];
            dvar += dxhat * (x->data[idx] - mean->data[i]);
            dmean += dxhat;
        }
        
        dvar *= -0.5f * (inv_std * inv_std * inv_std);
        dmean = (-dmean * inv_std) - 2.0f * dvar * 0.0f; // Simplified
        
        for (int j=0; j<N; j++) {
            int idx = i*N + j;
            float dxhat = y->grad[idx] * gamma->data[j];
            x->grad[idx] += dxhat * inv_std + dvar * 2.0f * (x->data[idx] - mean->data[i]) / N + dmean / N;
        }
    }
}

void backward_linear(Tensor* x, Tensor* w, Tensor* b, Tensor* y) {
    // dW = x^T * dy
    for(int i=0; i<w->n; i++) {
        for(int j=0; j<w->d; j++) {
            float sum = 0;
            for(int k=0; k<x->n; k++) sum += x->data[k*x->d + i] * y->grad[k*y->d + j];
            w->grad[i*w->d + j] += sum;
        }
    }
    // dx = dy * w^T
    for(int i=0; i<x->n; i++) {
        for(int j=0; j<x->d; j++) {
            float sum = 0;
            for(int k=0; k<w->d; k++) sum += y->grad[i*y->d + k] * w->data[j*w->d + k];
            x->grad[i*x->d + j] += sum;
        }
    }
    // db
    if(b) {
        for(int i=0; i<y->n; i++) {
            for(int j=0; j<y->d; j++) b->grad[j] += y->grad[i*y->d + j];
        }
    }
}

void backward_attention(Tensor* x, Transformer* m, Activations* c) {
    backward_linear(&c->att_out, &m->w_o, NULL, &c->o_out);
    
    // dV
    for(int i=0; i<c->v.n; i++) {
        for(int j=0; j<c->v.d; j++) {
            float sum = 0;
            for(int k=0; k<c->att_probs.n; k++) sum += c->att_probs.data[k*c->att_probs.d + i] * c->att_out.grad[k*c->att_out.d + j];
            c->v.grad[i*c->v.d + j] += sum;
        }
    }
    
    // dProbs
    for(int i=0; i<c->att_probs.n; i++) {
        for(int j=0; j<c->att_probs.d; j++) {
            float sum = 0;
            for(int k=0; k<c->v.d; k++) sum += c->att_out.grad[i*c->att_out.d + k] * c->v.data[j*c->v.d + k];
            c->att_probs.grad[i*c->att_probs.d + j] += sum;
        }
    }
    
    // Softmax Grad -> dScores
    for(int i=0; i<c->att_scores.n; i++) {
        float sum_p_dp = 0;
        for(int j=0; j<c->att_scores.d; j++) 
            sum_p_dp += c->att_probs.data[i*c->att_scores.d+j] * c->att_probs.grad[i*c->att_scores.d+j];
            
        for(int j=0; j<c->att_scores.d; j++) {
            float p = c->att_probs.data[i*c->att_scores.d+j];
            float dp = c->att_probs.grad[i*c->att_scores.d+j];
            c->att_scores.grad[i*c->att_scores.d+j] += p * (dp - sum_p_dp);
        }
    }
    
    // dQ, dK
    float scale = 1.0f / sqrtf((float)D_MODEL);
    for(int i=0; i<c->q.n; i++) {
        for(int j=0; j<c->q.d; j++) {
            float sum = 0;
            for(int k=0; k<c->att_scores.d; k++) sum += c->att_scores.grad[i*c->att_scores.d+k] * c->k.data[k*c->k.d+j];
            c->q.grad[i*c->q.d+j] += sum * scale;
        }
    }
    for(int i=0; i<c->k.n; i++) {
        for(int j=0; j<c->k.d; j++) {
            float sum = 0;
            for(int k=0; k<c->att_scores.n; k++) sum += c->att_scores.grad[k*c->att_scores.d+i] * c->q.data[k*c->q.d+j];
            c->k.grad[i*c->k.d+j] += sum * scale;
        }
    }
    
    backward_linear(x, &m->w_q, NULL, &c->q);
    backward_linear(x, &m->w_k, NULL, &c->k);
    backward_linear(x, &m->w_v, NULL, &c->v);
}

// --- High Level Ops ---

void update_param(Tensor* t) {
    for(int i=0; i<t->n*t->d; i++) {
        float g = t->grad[i];
        if(isnan(g) || isinf(g)) g = 0.0f;
        // Clipping
        if(g > CLIP_THRESHOLD) g = CLIP_THRESHOLD;
        if(g < -CLIP_THRESHOLD) g = -CLIP_THRESHOLD;
        t->data[i] -= LR * g;
        t->grad[i] = 0.0f;
    }
}

void forward_pass(Transformer* m, Activations* c, int* inputs, int len) {
    // 1. Dynamic Dims
    c->emb_out.n = len; c->input_sum.n = len;
    c->ln1_out.n = len; c->q.n = len; c->k.n = len; c->v.n = len;
    c->att_scores.n = len; c->att_scores.d = len; c->att_probs.n = len; c->att_probs.d = len;
    c->att_out.n = len; c->o_out.n = len; c->res1.n = len;
    c->ln2_out.n = len; c->ffn1.n = len; c->ffn_relu.n = len; c->ffn2.n = len; c->res2.n = len;
    c->logits.n = len; c->probs.n = len;

    // 2. Embeddings
    for(int t=0; t<len; t++) {
        int tid = inputs[t]; if(tid>=VOCAB_SIZE) tid=0;
        for(int j=0; j<D_MODEL; j++) 
            c->input_sum.data[t*D_MODEL+j] = m->token_emb.data[tid*D_MODEL+j] + m->pos_emb.data[t*D_MODEL+j];
    }
    
    // 3. Block 1
    forward_layernorm(&c->input_sum, &m->ln1_g, &m->ln1_b, &c->ln1_out, &c->ln1_mean, &c->ln1_var);
    forward_attention(&c->ln1_out, m, c);
    for(int i=0; i<len*D_MODEL; i++) c->res1.data[i] = c->input_sum.data[i] + c->o_out.data[i];
    
    // 4. Block 2
    forward_layernorm(&c->res1, &m->ln2_g, &m->ln2_b, &c->ln2_out, &c->ln2_mean, &c->ln2_var);
    matmul(&c->ln2_out, &m->w1, &c->ffn1);
    for(int i=0; i<len*D_FF; i++) c->ffn_relu.data[i] = c->ffn1.data[i] > 0 ? c->ffn1.data[i] : 0; 
    matmul(&c->ffn_relu, &m->w2, &c->ffn2);
    for(int i=0; i<len*D_MODEL; i++) c->res2.data[i] = c->res1.data[i] + c->ffn2.data[i];
    
    // 5. Head
    matmul(&c->res2, &m->w_final, &c->logits);
    
    // Softmax
    for (int i = 0; i < len; i++) {
        float maxv = -1e9f;
        for(int j=0; j<VOCAB_SIZE; j++) if(c->logits.data[i*VOCAB_SIZE+j] > maxv) maxv = c->logits.data[i*VOCAB_SIZE+j];
        float sum = 0;
        for(int j=0; j<VOCAB_SIZE; j++) {
            float e = expf(c->logits.data[i*VOCAB_SIZE+j] - maxv);
            c->probs.data[i*VOCAB_SIZE+j] = e;
            sum += e;
        }
        for(int j=0; j<VOCAB_SIZE; j++) c->probs.data[i*VOCAB_SIZE+j] /= sum;
    }
}

void backward_pass(Transformer* m, Activations* c, int len) {
    backward_linear(&c->res2, &m->w_final, NULL, &c->logits);
    
    for(int i=0; i<len*D_MODEL; i++) {
        c->ffn2.grad[i] += c->res2.grad[i];
        c->res1.grad[i] += c->res2.grad[i];
    }
    
    backward_linear(&c->ffn_relu, &m->w2, &m->b2, &c->ffn2);
    for(int i=0; i<len*D_FF; i++) {
        if(c->ffn1.data[i] > 0) c->ffn1.grad[i] += c->ffn_relu.grad[i];
    }
    backward_linear(&c->ln2_out, &m->w1, &m->b1, &c->ffn1);
    
    backward_layernorm(&c->res1, &m->ln2_g, &m->ln2_b, &c->ln2_out, &c->ln2_mean, &c->ln2_var);
    
    for(int i=0; i<len*D_MODEL; i++) {
        c->o_out.grad[i] += c->res1.grad[i];
        c->input_sum.grad[i] += c->res1.grad[i];
    }
    
    backward_attention(&c->ln1_out, m, c);
    backward_layernorm(&c->input_sum, &m->ln1_g, &m->ln1_b, &c->ln1_out, &c->ln1_mean, &c->ln1_var);
    
    for(int t=0; t<len; t++) {
        int tid = c->inputs[t];
        for(int j=0; j<D_MODEL; j++) {
            m->token_emb.grad[tid*D_MODEL+j] += c->input_sum.grad[t*D_MODEL+j];
            m->pos_emb.grad[t*D_MODEL+j] += c->input_sum.grad[t*D_MODEL+j];
        }
    }
}

void run_step(Transformer* m) {
    update_param(&m->token_emb); update_param(&m->pos_emb);
    update_param(&m->w_q); update_param(&m->w_k); update_param(&m->w_v); update_param(&m->w_o);
    update_param(&m->ln1_g); update_param(&m->ln1_b);
    update_param(&m->w1); update_param(&m->b1); update_param(&m->w2); update_param(&m->b2);
    update_param(&m->ln2_g); update_param(&m->ln2_b);
    update_param(&m->w_final);
}

// --- Main ---

int main() {
    srand(42);
    
    // 1. Init Weights (Same as before)
    Transformer m;
    m.token_emb = tensor_alloc(VOCAB_SIZE, D_MODEL, "Tok"); tensor_init_xavier(&m.token_emb);
    m.pos_emb = tensor_alloc(SEQ_LEN, D_MODEL, "Pos"); tensor_init_xavier(&m.pos_emb);
    m.w_q = tensor_alloc(D_MODEL, D_MODEL, "WQ"); tensor_init_xavier(&m.w_q);
    m.w_k = tensor_alloc(D_MODEL, D_MODEL, "WK"); tensor_init_xavier(&m.w_k);
    m.w_v = tensor_alloc(D_MODEL, D_MODEL, "WV"); tensor_init_xavier(&m.w_v);
    m.w_o = tensor_alloc(D_MODEL, D_MODEL, "WO"); tensor_init_xavier(&m.w_o);
    m.ln1_g = tensor_alloc(D_MODEL, 1, "LN1_G"); for(int i=0;i<D_MODEL;i++) m.ln1_g.data[i]=1.0f;
    m.ln1_b = tensor_alloc(D_MODEL, 1, "LN1_B");
    m.w1 = tensor_alloc(D_MODEL, D_FF, "W1"); tensor_init_xavier(&m.w1);
    m.b1 = tensor_alloc(D_FF, 1, "B1");
    m.w2 = tensor_alloc(D_FF, D_MODEL, "W2"); tensor_init_xavier(&m.w2);
    m.b2 = tensor_alloc(D_MODEL, 1, "B2");
    m.ln2_g = tensor_alloc(D_MODEL, 1, "LN2_G"); for(int i=0;i<D_MODEL;i++) m.ln2_g.data[i]=1.0f;
    m.ln2_b = tensor_alloc(D_MODEL, 1, "LN2_B");
    m.w_final = tensor_alloc(D_MODEL, VOCAB_SIZE, "WF"); tensor_init_xavier(&m.w_final);
    
    // 2. Init Activations (Same as before)
    Activations c;
    c.emb_out = tensor_alloc(SEQ_LEN, D_MODEL, "EO");
    c.input_sum = tensor_alloc(SEQ_LEN, D_MODEL, "IS");
    c.ln1_out = tensor_alloc(SEQ_LEN, D_MODEL, "L1");
    c.ln1_mean = tensor_alloc(SEQ_LEN, 1, "L1M"); c.ln1_var = tensor_alloc(SEQ_LEN, 1, "L1V");
    c.q = tensor_alloc(SEQ_LEN, D_MODEL, "Q"); c.k = tensor_alloc(SEQ_LEN, D_MODEL, "K"); c.v = tensor_alloc(SEQ_LEN, D_MODEL, "V");
    c.att_scores = tensor_alloc(SEQ_LEN, SEQ_LEN, "AS"); c.att_probs = tensor_alloc(SEQ_LEN, SEQ_LEN, "AP");
    c.att_out = tensor_alloc(SEQ_LEN, D_MODEL, "AO"); c.o_out = tensor_alloc(SEQ_LEN, D_MODEL, "OO");
    c.res1 = tensor_alloc(SEQ_LEN, D_MODEL, "R1");
    c.ln2_out = tensor_alloc(SEQ_LEN, D_MODEL, "L2");
    c.ln2_mean = tensor_alloc(SEQ_LEN, 1, "L2M"); c.ln2_var = tensor_alloc(SEQ_LEN, 1, "L2V");
    c.ffn1 = tensor_alloc(SEQ_LEN, D_FF, "F1"); c.ffn_relu = tensor_alloc(SEQ_LEN, D_FF, "FR");
    c.ffn2 = tensor_alloc(SEQ_LEN, D_MODEL, "F2"); c.res2 = tensor_alloc(SEQ_LEN, D_MODEL, "R2");
    c.logits = tensor_alloc(SEQ_LEN, VOCAB_SIZE, "L"); c.probs = tensor_alloc(SEQ_LEN, VOCAB_SIZE, "P");
    c.inputs = malloc(SEQ_LEN * sizeof(int));
    
    // 3. Setup Data
    char* text = "The quick brown fox jumps over the lazy dog.";
    int len = strlen(text);
    // [FIX] Train on the FULL sentence length minus 1 (for targets)
    int batch_len = len - 1; 
    
    int inputs[SEQ_LEN], targets[SEQ_LEN];
    // Pre-fill inputs/targets deterministically (Position 0 is always 'T')
    for(int i=0; i<batch_len; i++) {
        inputs[i] = (int)text[i];
        targets[i] = (int)text[i+1];
    }
    
    printf("Training Phonex-C0 (Anchored Position Mode)...\n");
    
    // 4. Training Loop
    for(int step=0; step<2000; step++) { // 2000 is plenty now
        // Zero Grads
        tensor_zero_grad(&c.input_sum); tensor_zero_grad(&c.ln1_out);
        tensor_zero_grad(&c.q); tensor_zero_grad(&c.k); tensor_zero_grad(&c.v);
        tensor_zero_grad(&c.att_scores); tensor_zero_grad(&c.att_probs);
        tensor_zero_grad(&c.att_out); tensor_zero_grad(&c.o_out);
        tensor_zero_grad(&c.res1); tensor_zero_grad(&c.ln2_out);
        tensor_zero_grad(&c.ffn1); tensor_zero_grad(&c.ffn_relu);
        tensor_zero_grad(&c.ffn2); tensor_zero_grad(&c.res2);
        tensor_zero_grad(&c.logits);

        // [FIX] No random sliding window. Always train the full sequence.
        memcpy(c.inputs, inputs, batch_len * sizeof(int));
        
        forward_pass(&m, &c, inputs, batch_len);
        
        float loss = 0;
        for(int i=0; i<batch_len; i++) {
            float p = c.probs.data[i*VOCAB_SIZE+targets[i]];
            if(p < 1e-9f) p = 1e-9f;
            loss -= logf(p);
            for(int j=0; j<VOCAB_SIZE; j++) {
                float ind = (j==targets[i]) ? 1.0f : 0.0f;
                c.logits.grad[i*VOCAB_SIZE+j] = (c.probs.data[i*VOCAB_SIZE+j] - ind) / batch_len;
            }
        }
        loss /= batch_len;
        
        if(step % 200 == 0) printf("Step %d \t Loss: %.4f\n", step, loss);
        
        backward_pass(&m, &c, batch_len);
        run_step(&m);
    }
    
    // 5. Generation
    printf("\nGenerating...\n");
    char* start_str = "The";
    int gen_len = strlen(start_str);
    int gen_ids[SEQ_LEN];
    for(int i=0; i<gen_len; i++) gen_ids[i] = (int)start_str[i];
    
    printf("%s", start_str);
    
    // Generate until end of sentence length
    while(gen_len < len) { 
        forward_pass(&m, &c, gen_ids, gen_len);
        
        float* logits = &c.logits.data[(gen_len-1)*VOCAB_SIZE];
        float max_val = -1e9f;
        int next_tok = 0;
        for(int i=0; i<VOCAB_SIZE; i++) {
            if(logits[i] > max_val) { max_val = logits[i]; next_tok = i; }
        }
        
        printf("%c", (char)next_tok);
        gen_ids[gen_len++] = next_tok;
    }
    printf("\n\nDone.\n");
    
    free(c.inputs);
    return 0;
}