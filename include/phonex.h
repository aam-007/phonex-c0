#ifndef PHONEX_H
#define PHONEX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// --- Hyperparameters ---
#define D_MODEL 16
#define D_FF 32
#define SEQ_LEN 16
#define VOCAB_SIZE 128
#define EPSILON 1e-9f

// --- Data Structures ---

typedef struct {
    float* data;
    float* grad;
    int n; // Rows (Batch/Seq)
    int d; // Cols (Dim)
} Tensor;

typedef struct {
    Tensor token_emb;
    Tensor pos_emb;
    
    // Attention
    Tensor w_q, w_k, w_v, w_o;
    
    // Layer Norm 1
    Tensor ln1_g, ln1_b;
    
    // FFN
    Tensor w1, b1, w2, b2;
    
    // Layer Norm 2
    Tensor ln2_g, ln2_b;
    
    // Head
    Tensor w_final;
} Transformer;

typedef struct {
    // Cache for backprop
    Tensor emb_out, input_sum;
    Tensor ln1_out, ln1_mean, ln1_var;
    Tensor q, k, v, att_scores, att_probs, att_out, o_out, res1;
    Tensor ln2_out, ln2_mean, ln2_var;
    Tensor ffn1, ffn_relu, ffn2, res2;
    Tensor logits, probs;
    int* inputs;
} Activations;

// --- Prototypes ---

// tensor.c
Tensor tensor_alloc(int n, int d);
void tensor_free(Tensor* t);
void tensor_init_xavier(Tensor* t);
void tensor_zero_grad(Tensor* t);
void matmul(Tensor* A, Tensor* B, Tensor* C);
void matmul_at_b_accum(Tensor* A, Tensor* B, Tensor* C_grad);
void matmul_c_bt_accum(Tensor* C, Tensor* B, Tensor* A_grad);

// attention.c
void forward_attention(Tensor* x, Transformer* m, Activations* c);
void backward_attention(Tensor* x, Transformer* m, Activations* c);

// layernorm.c
void forward_layernorm(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* out, Tensor* mean, Tensor* var);
void backward_layernorm(Tensor* x, Tensor* gamma, Tensor* beta, Tensor* y, Tensor* mean, Tensor* var);

// ffw.c
void forward_ffn(Tensor* x, Transformer* m, Activations* c);
void backward_ffn(Tensor* x, Transformer* m, Activations* c);
void backward_linear(Tensor* x, Tensor* w, Tensor* b, Tensor* y_out); // Shared helper

// loss.c
void softmax(Tensor* input, Tensor* output);
float cross_entropy_loss(Activations* c, int* targets);

// optimizer.c
void model_step(Transformer* m, float lr);

// tokenizer.c
void tokenize_input(const char* text, int* out_ids, int len);
char detokenize_id(int id);

// main utils
void init_model(Transformer* m);
Activations alloc_activations(int T);
void reset_activations_grad(Activations* c);

#endif