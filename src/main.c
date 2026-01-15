// Initialization, file reading, training loop, and generation

#include "../include/phonex.h"

void forward_embeddings(int* inputs, Tensor* tok_emb, Tensor* pos_emb, Tensor* out) {
    for (int t = 0; t < out->n; t++) {
        int token_id = inputs[t];
        for (int i = 0; i < out->d; i++) {
            out->data[t * out->d + i] = 
                tok_emb->data[token_id * tok_emb->d + i] + 
                pos_emb->data[t * pos_emb->d + i];
        }
    }
}

void model_forward(Transformer* m, Activations* c, int* inputs, int len) {
    memcpy(c->inputs, inputs, len * sizeof(int));
    
    // [FIX 1] Set dynamic length for the tensors used in this pass
    c->emb_out.n = len; 
    c->input_sum.n = len; 
    
    forward_embeddings(inputs, &m->token_emb, &m->pos_emb, &c->input_sum);
    
    // Block 1: Norm -> Attn -> Res
    forward_layernorm(&c->input_sum, &m->ln1_g, &m->ln1_b, &c->ln1_out, &c->ln1_mean, &c->ln1_var);
    
    c->att_scores.n = len; c->att_scores.d = len;
    c->att_probs.n = len; c->att_probs.d = len;
    
    forward_attention(&c->ln1_out, m, c);
    
    // [FIX 2] Ensure residual tensors match length
    c->res1.n = len; 
    for (int i = 0; i < len * D_MODEL; i++) 
        c->res1.data[i] = c->input_sum.data[i] + c->o_out.data[i];
        
    // Block 2: Norm -> FFN -> Res
    forward_layernorm(&c->res1, &m->ln2_g, &m->ln2_b, &c->ln2_out, &c->ln2_mean, &c->ln2_var);
    
    forward_ffn(&c->ln2_out, m, c);
    
    c->res2.n = len;
    for (int i = 0; i < len * D_MODEL; i++) 
        c->res2.data[i] = c->res1.data[i] + c->ffn2.data[i];
        
    // Output
    matmul(&c->res2, &m->w_final, &c->logits);
    softmax(&c->logits, &c->probs);
}

void model_backward(Transformer* m, Activations* c) {
    backward_linear(&c->res2, &m->w_final, NULL, &c->logits);
    
    for (int i = 0; i < c->res2.n * D_MODEL; i++) {
        c->ffn2.grad[i] += c->res2.grad[i];
        c->res1.grad[i] += c->res2.grad[i];
    }
    
    backward_ffn(&c->ln2_out, m, c);
    backward_layernorm(&c->res1, &m->ln2_g, &m->ln2_b, &c->ln2_out, &c->ln2_mean, &c->ln2_var);
    
    for (int i = 0; i < c->res1.n * D_MODEL; i++) {
        c->o_out.grad[i] += c->res1.grad[i];
        c->input_sum.grad[i] += c->res1.grad[i];
    }
    
    backward_attention(&c->ln1_out, m, c);
    backward_layernorm(&c->input_sum, &m->ln1_g, &m->ln1_b, &c->ln1_out, &c->ln1_mean, &c->ln1_var);
    
    for (int t = 0; t < c->emb_out.n; t++) {
        int tok_id = c->inputs[t];
        for (int j = 0; j < D_MODEL; j++) {
            m->token_emb.grad[tok_id * D_MODEL + j] += c->input_sum.grad[t * D_MODEL + j];
            m->pos_emb.grad[t * D_MODEL + j] += c->input_sum.grad[t * D_MODEL + j];
        }
    }
}

// Memory and Init Helpers 
void init_model(Transformer* m) {
    m->token_emb = tensor_alloc(VOCAB_SIZE, D_MODEL); tensor_init_xavier(&m->token_emb);
    m->pos_emb = tensor_alloc(SEQ_LEN, D_MODEL); tensor_init_xavier(&m->pos_emb);
    m->w_q = tensor_alloc(D_MODEL, D_MODEL); tensor_init_xavier(&m->w_q);
    m->w_k = tensor_alloc(D_MODEL, D_MODEL); tensor_init_xavier(&m->w_k);
    m->w_v = tensor_alloc(D_MODEL, D_MODEL); tensor_init_xavier(&m->w_v);
    m->w_o = tensor_alloc(D_MODEL, D_MODEL); tensor_init_xavier(&m->w_o);
    m->ln1_g = tensor_alloc(D_MODEL, 1); for(int i=0;i<D_MODEL;i++) m->ln1_g.data[i]=1.0f;
    m->ln1_b = tensor_alloc(D_MODEL, 1);
    m->w1 = tensor_alloc(D_MODEL, D_FF); tensor_init_xavier(&m->w1);
    m->b1 = tensor_alloc(D_FF, 1);
    m->w2 = tensor_alloc(D_FF, D_MODEL); tensor_init_xavier(&m->w2);
    m->b2 = tensor_alloc(D_MODEL, 1);
    m->ln2_g = tensor_alloc(D_MODEL, 1); for(int i=0;i<D_MODEL;i++) m->ln2_g.data[i]=1.0f;
    m->ln2_b = tensor_alloc(D_MODEL, 1);
    m->w_final = tensor_alloc(D_MODEL, VOCAB_SIZE); tensor_init_xavier(&m->w_final);
}

Activations alloc_activations(int T) {
    Activations c;
    c.emb_out = tensor_alloc(T, D_MODEL);
    c.input_sum = tensor_alloc(T, D_MODEL);
    c.ln1_out = tensor_alloc(T, D_MODEL);
    c.ln1_mean = tensor_alloc(T, 1); c.ln1_var = tensor_alloc(T, 1);
    c.q = tensor_alloc(T, D_MODEL); c.k = tensor_alloc(T, D_MODEL); c.v = tensor_alloc(T, D_MODEL);
    c.att_scores = tensor_alloc(T, T); c.att_probs = tensor_alloc(T, T); c.att_out = tensor_alloc(T, D_MODEL);
    c.o_out = tensor_alloc(T, D_MODEL); c.res1 = tensor_alloc(T, D_MODEL);
    c.ln2_out = tensor_alloc(T, D_MODEL);
    c.ln2_mean = tensor_alloc(T, 1); c.ln2_var = tensor_alloc(T, 1);
    c.ffn1 = tensor_alloc(T, D_FF); c.ffn_relu = tensor_alloc(T, D_FF); c.ffn2 = tensor_alloc(T, D_MODEL); c.res2 = tensor_alloc(T, D_MODEL);
    c.logits = tensor_alloc(T, VOCAB_SIZE); c.probs = tensor_alloc(T, VOCAB_SIZE);
    c.inputs = malloc(T * sizeof(int));
    return c;
}

void reset_activations_grad(Activations* c) {
    tensor_zero_grad(&c->input_sum);
    tensor_zero_grad(&c->ln1_out);
    tensor_zero_grad(&c->q); tensor_zero_grad(&c->k); tensor_zero_grad(&c->v);
    tensor_zero_grad(&c->att_scores); tensor_zero_grad(&c->att_probs);
    tensor_zero_grad(&c->att_out); tensor_zero_grad(&c->o_out);
    tensor_zero_grad(&c->res1);
    tensor_zero_grad(&c->ln2_out);
    tensor_zero_grad(&c->ffn1); tensor_zero_grad(&c->ffn_relu);
    tensor_zero_grad(&c->ffn2); tensor_zero_grad(&c->res2);
    tensor_zero_grad(&c->logits);
}

void generate(Transformer* m, char* start_str, int max_new_tokens) {
    int input_ids[SEQ_LEN];
    int len = strlen(start_str);
    tokenize_input(start_str, input_ids, len);
    
    printf("\n[Generation]: %s", start_str);
    Activations c = alloc_activations(SEQ_LEN);
    
    for (int k = 0; k < max_new_tokens; k++) {
        if (len >= SEQ_LEN) break;
        model_forward(m, &c, input_ids, len);
        float* logits = &c.logits.data[(len-1) * VOCAB_SIZE];
        int next_token = 0;
        float max_logit = -1e9f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (logits[i] > max_logit) { max_logit = logits[i]; next_token = i; }
        }
        printf("%c", detokenize_id(next_token));
        input_ids[len++] = next_token;
    }
    printf("\n");

}

int main() {
    srand(time(NULL));
    Transformer model;
    init_model(&model);
    Activations cache = alloc_activations(SEQ_LEN);
    
    // Load Data
    FILE* f = fopen("data/tiny_corpus.txt", "r");
    char text[1024];
    if (f) {
        if (!fgets(text, 1024, f)) strcpy(text, "hello world hello world");
        fclose(f);
    } else {
        strcpy(text, "hello world hello world");
    }
    
    int data_len = strlen(text);
    // Ensure we don't overrun if file is too short
    if (data_len < SEQ_LEN + 1) {
        strcpy(text, "The quick brown fox jumps over the lazy dog. ");
        data_len = strlen(text);
    }

    int inputs[SEQ_LEN], targets[SEQ_LEN];
    
    printf("Training on: '%s' (len: %d)\n", text, data_len);
    
    for (int step = 0; step < 5000; step++) {
        // [FIX 3] Zero gradients BEFORE forward pass (or at start of step)
        reset_activations_grad(&cache);

        int start_idx = rand() % (data_len - 8 - 1);
        for (int i = 0; i < 8; i++) {
            inputs[i] = (int)text[start_idx + i];
            targets[i] = (int)text[start_idx + i + 1];
        }
        
        model_forward(&model, &cache, inputs, 8);
        
        float loss = cross_entropy_loss(&cache, targets);
        if (step % 500 == 0) printf("%d\t%.4f\n", step, loss);
        
        // DO NOT reset grads here, or you wipe the loss gradient!
        model_backward(&model, &cache);
        model_step(&model, 0.01f);
    }
    
    generate(&model, "The", 10); // Generate a bit more
    return 0;
}