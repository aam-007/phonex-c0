// Initialization, file reading, training loop, and generation

#include "../include/phonex.h"

// --- Forward Pass Helpers ---

void forward_embeddings(int* inputs, Tensor* tok_emb, Tensor* pos_emb, Tensor* out) {
    for (int t = 0; t < out->n; t++) {
        int token_id = inputs[t];
        // SAFETY CHECK: Prevent segfault if garbage data gets in
        if (token_id < 0 || token_id >= VOCAB_SIZE) token_id = 0; 

        for (int i = 0; i < out->d; i++) {
            out->data[t * out->d + i] = 
                tok_emb->data[token_id * tok_emb->d + i] + 
                pos_emb->data[t * pos_emb->d + i];
        }
    }
}

void model_forward(Transformer* m, Activations* c, int* inputs, int len) {
    // 1. Update Tensor Dimensions to match current batch length
    c->emb_out.n = len;
    c->input_sum.n = len;
    c->ln1_out.n = len;
    c->q.n = len; c->k.n = len; c->v.n = len;
    c->att_scores.n = len; c->att_scores.d = len; // NxN map
    c->att_probs.n = len; c->att_probs.d = len;
    c->att_out.n = len;
    c->o_out.n = len;
    c->res1.n = len;
    c->ln2_out.n = len;
    c->ffn1.n = len; c->ffn_relu.n = len; 
    c->ffn2.n = len;
    c->res2.n = len;
    c->logits.n = len;
    c->probs.n = len;

    // 2. Copy inputs
    memcpy(c->inputs, inputs, len * sizeof(int));
    
    // 3. Run Layers
    forward_embeddings(inputs, &m->token_emb, &m->pos_emb, &c->input_sum);
    
    // Block 1
    forward_layernorm(&c->input_sum, &m->ln1_g, &m->ln1_b, &c->ln1_out, &c->ln1_mean, &c->ln1_var);
    forward_attention(&c->ln1_out, m, c);
    
    // Res 1
    for (int i = 0; i < len * D_MODEL; i++) 
        c->res1.data[i] = c->input_sum.data[i] + c->o_out.data[i];
        
    // Block 2
    forward_layernorm(&c->res1, &m->ln2_g, &m->ln2_b, &c->ln2_out, &c->ln2_mean, &c->ln2_var);
    forward_ffn(&c->ln2_out, m, c);
    
    // Res 2
    for (int i = 0; i < len * D_MODEL; i++) 
        c->res2.data[i] = c->res1.data[i] + c->ffn2.data[i];
        
    // Output
    matmul(&c->res2, &m->w_final, &c->logits);
    softmax(&c->logits, &c->probs);
}

void model_backward(Transformer* m, Activations* c) {
    // 1. Output Head
    backward_linear(&c->res2, &m->w_final, NULL, &c->logits);
    
    // 2. Split Gradient at Res2
    for (int i = 0; i < c->res2.n * D_MODEL; i++) {
        c->ffn2.grad[i] += c->res2.grad[i];
        c->res1.grad[i] += c->res2.grad[i];
    }
    
    // 3. Block 2 Backward
    backward_ffn(&c->ln2_out, m, c);
    backward_layernorm(&c->res1, &m->ln2_g, &m->ln2_b, &c->ln2_out, &c->ln2_mean, &c->ln2_var);
    
    // 4. Split Gradient at Res1
    for (int i = 0; i < c->res1.n * D_MODEL; i++) {
        c->o_out.grad[i] += c->res1.grad[i];
        c->input_sum.grad[i] += c->res1.grad[i];
    }
    
    // 5. Block 1 Backward
    backward_attention(&c->ln1_out, m, c);
    backward_layernorm(&c->input_sum, &m->ln1_g, &m->ln1_b, &c->ln1_out, &c->ln1_mean, &c->ln1_var);
    
    // 6. Embeddings Accumulation
    for (int t = 0; t < c->emb_out.n; t++) {
        int tok_id = c->inputs[t];
        if (tok_id < 0 || tok_id >= VOCAB_SIZE) continue; // Safety
        
        for (int j = 0; j < D_MODEL; j++) {
            m->token_emb.grad[tok_id * D_MODEL + j] += c->input_sum.grad[t * D_MODEL + j];
            m->pos_emb.grad[t * D_MODEL + j] += c->input_sum.grad[t * D_MODEL + j];
        }
    }
}

// --- Init & Memory ---

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
    
    c.ffn1 = tensor_alloc(T, D_FF); c.ffn_relu = tensor_alloc(T, D_FF); 
    c.ffn2 = tensor_alloc(T, D_MODEL); c.res2 = tensor_alloc(T, D_MODEL);
    
    c.logits = tensor_alloc(T, VOCAB_SIZE); c.probs = tensor_alloc(T, VOCAB_SIZE);
    
    c.inputs = malloc(T * sizeof(int));
    return c;
}

void reset_activations_grad(Activations* c) {
  
    
    c->input_sum.n = SEQ_LEN; tensor_zero_grad(&c->input_sum);
    c->ln1_out.n = SEQ_LEN; tensor_zero_grad(&c->ln1_out);
    
    c->q.n = SEQ_LEN; tensor_zero_grad(&c->q);
    c->k.n = SEQ_LEN; tensor_zero_grad(&c->k);
    c->v.n = SEQ_LEN; tensor_zero_grad(&c->v);
    
    // Attention maps are special (TxT)
    c->att_scores.n = SEQ_LEN; c->att_scores.d = SEQ_LEN; tensor_zero_grad(&c->att_scores);
    c->att_probs.n = SEQ_LEN; c->att_probs.d = SEQ_LEN; tensor_zero_grad(&c->att_probs);
    
    c->att_out.n = SEQ_LEN; tensor_zero_grad(&c->att_out);
    c->o_out.n = SEQ_LEN; tensor_zero_grad(&c->o_out);
    c->res1.n = SEQ_LEN; tensor_zero_grad(&c->res1);
    
    c->ln2_out.n = SEQ_LEN; tensor_zero_grad(&c->ln2_out);
    c->ffn1.n = SEQ_LEN; tensor_zero_grad(&c->ffn1);
    c->ffn_relu.n = SEQ_LEN; tensor_zero_grad(&c->ffn_relu);
    c->ffn2.n = SEQ_LEN; tensor_zero_grad(&c->ffn2);
    c->res2.n = SEQ_LEN; tensor_zero_grad(&c->res2);
    
    c->logits.n = SEQ_LEN; tensor_zero_grad(&c->logits);
}

void generate(Transformer* m, char* start_str, int max_new_tokens) {
    int input_ids[SEQ_LEN];
    int len = strlen(start_str);
    tokenize_input(start_str, input_ids, len);
    
    printf("\n[Generation]: %s", start_str);
    Activations c = alloc_activations(SEQ_LEN);
    
    for (int k = 0; k < max_new_tokens; k++) {
        if (len >= SEQ_LEN) break;
        
        // Forward pass with current length
        model_forward(m, &c, input_ids, len);
        
        // Sampling
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
    // Leak: c is not freed here, fine for toy example
}

int main() {
    srand(time(NULL));
    Transformer model;
    init_model(&model);
    Activations cache = alloc_activations(SEQ_LEN);
    
    // Load Data or Default
    FILE* f = fopen("data/tiny_corpus.txt", "r");
    char text[1024];
    if (f) {
        if (!fgets(text, 1024, f)) strcpy(text, "hello world hello world");
        fclose(f);
    } else {
        strcpy(text, "hello world hello world");
    }
    
    int data_len = strlen(text);
    if (data_len < SEQ_LEN + 1) {
        // Fallback if file is empty/too short
        strcpy(text, "The quick brown fox jumps over the lazy dog. ");
        data_len = strlen(text);
    }
    
    int inputs[SEQ_LEN];
    int targets[SEQ_LEN];
    
    printf("Training on: '%s' (len: %d)\n", text, data_len);
    printf("Steps\tLoss\n");
    
    for (int step = 0; step < 5000; step++) {
        // 1. Zero Gradients explicitly covering FULL memory
        reset_activations_grad(&cache);
        
        // 2. Prepare Batch
        int start_idx = rand() % (data_len - 8 - 1);
        for (int i = 0; i < 8; i++) {
            inputs[i] = (int)text[start_idx + i];
            targets[i] = (int)text[start_idx + i + 1];
        }
        
        // 3. Forward
        model_forward(&model, &cache, inputs, 8);
        
        // 4. Loss
        float loss = cross_entropy_loss(&cache, targets);
        if (step % 500 == 0) printf("%d\t%.4f\n", step, loss);
        
        // 5. Backward
        model_backward(&model, &cache);
        
        // 6. Update
        model_step(&model, 0.001f);
    }
    
    generate(&model, "The", 15);
    generate(&model, "fox", 15);
    
    return 0;
}