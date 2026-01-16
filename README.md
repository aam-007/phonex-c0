# Phonex-C0: Persistent Transformer in Pure C

A minimal, educational implementation of a Transformer neural network written entirely in C99 with binary model persistence, gradient-based training, and text generation capabilities.

## Overview

Phonex-C0 is a from-scratch implementation of the Transformer architecture that demonstrates core deep learning concepts without external dependencies. The project includes:

- **Self-contained implementation**: No PyTorch, TensorFlow, or other ML frameworks
- **Binary model persistence**: Save and load trained weights
- **Complete training loop**: Forward pass, backpropagation, and gradient descent
- **Text generation**: Autoregressive sampling for sequence completion
- **Educational focus**: Clear, readable code optimized for learning

This implementation is designed for understanding how Transformers work at a fundamental level, making it ideal for students, researchers, and anyone curious about the internals of modern language models.

## Architecture

The model implements a simplified Transformer decoder with:

- **Token embeddings** (128 vocab × 16 dimensions)
- **Positional embeddings** (64 sequence length × 16 dimensions)
- **Single attention layer** with Query, Key, Value projections
- **Layer normalization** (pre-attention and pre-FFN)
- **Feed-forward network** (16 → 32 → 16 dimensions)
- **Residual connections** around attention and FFN blocks
- **Final linear projection** to vocabulary logits

### Hyperparameters

```c
D_MODEL = 16        // Model dimensionality
D_FF = 32           // Feed-forward hidden size
SEQ_LEN = 64        // Maximum sequence length
VOCAB_SIZE = 128    // ASCII character vocabulary
EPSILON = 1e-5      // Layer norm stability
CLIP_THRESHOLD = 5.0 // Gradient clipping
LR = 0.01           // Learning rate
```

## Features

### Core Components

1. **Tensor Engine**
   - Dynamic tensor allocation with gradient tracking
   - Xavier/Glorot weight initialization
   - Matrix multiplication primitives
   - Automatic gradient zeroing

2. **Persistence Layer**
   - Binary serialization of model weights
   - Efficient save/load operations
   - Dimension validation on load
   - Error handling for I/O operations

3. **Forward Operations**
   - Layer normalization with mean/variance caching
   - Scaled dot-product attention with causal masking
   - Softmax with numerical stability (max subtraction)
   - ReLU activation functions
   - Residual connections

4. **Backward Operations**
   - Reverse-mode automatic differentiation
   - Layer norm gradients with careful accumulation
   - Attention backpropagation (including softmax Jacobian)
   - Linear layer weight and input gradients
   - Gradient clipping for training stability

5. **Optimizer**
   - Vanilla SGD with gradient clipping
   - NaN/Inf gradient detection and handling
   - Configurable learning rate

## Installation

### Prerequisites

- GCC or any C99-compliant compiler
- Standard C math library (`libm`)
- POSIX-compatible system (Linux, macOS, Unix)

### Compilation

```bash
gcc -O3 c0p.c -lm -o c0p
```

**Compiler flags:**
- `-O3`: Maximum optimization for performance
- `-lm`: Link math library for `sqrt`, `exp`, `log` functions
- `-o c0p`: Output executable name

### Optional: Debug Build

```bash
gcc -g -Wall -Wextra c0p.c -lm -o c0p_debug
```

## Usage

### Training Mode

Train a new model from scratch and save weights to `model.bin`:

```bash
./c0p train
```

**Training process:**
1. Initializes model with Xavier-initialized weights
2. Creates training data from hardcoded text sequence
3. Runs 2000 training iterations with gradient descent
4. Prints loss every 200 steps
5. Saves final model to `model.bin`

**Example output:**
```
[TRAIN] Starting...
Step 0    Loss: 4.8532
Step 200  Loss: 2.3145
Step 400  Loss: 1.5673
...
Step 1800 Loss: 0.4521
[IO] Saving model to model.bin...
[IO] Save complete.
```

### Inference Mode

Load a trained model and generate text:

```bash
./c0p infer
```

**Generation process:**
1. Loads weights from `model.bin`
2. Seeds generation with prefix "The"
3. Autoregressively samples tokens using greedy decoding
4. Stops at sequence end or period character

**Example output:**
```
[IO] Loading model from model.bin...
[IO] Load complete.

[INFER] Generating...
The quick brown fox jumps over the lazy dog.
```

## Implementation Details

### Data Flow

```
Input Tokens → Token Embedding ┐
                               ├→ Add → LayerNorm → Attention → Add → LayerNorm → FFN → Add → Output Logits
Position IDs → Position Embed ┘                      ↑                  ↑
                                                      │                  │
                                                   Residual          Residual
```

### Attention Mechanism

The self-attention implementation uses causal masking to prevent positions from attending to future tokens:

```c
for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
        if (j > i) {
            scores[i][j] = -1e9;  // Mask future positions
        } else {
            scores[i][j] = dot(Q[i], K[j]) / sqrt(d_model);
        }
    }
}
```

### Training Algorithm

1. **Forward Pass**: Compute activations and cache intermediate values
2. **Loss Calculation**: Cross-entropy between predictions and targets
3. **Backward Pass**: Compute gradients via backpropagation
4. **Parameter Update**: Apply clipped gradients with SGD

```c
// Gradient descent update
param -= learning_rate * clip(gradient, threshold)
```

### Memory Layout

All tensors use row-major format:
```
Tensor[n×d] → data[n*d] where data[i*d + j] = element at (i,j)
```

## File Format

The `model.bin` file stores weights in binary format:

```
[n:int32][d:int32][data:float32×(n×d)] // token_emb
[n:int32][d:int32][data:float32×(n×d)] // pos_emb
[n:int32][d:int32][data:float32×(n×d)] // w_q
...
```

Each tensor is preceded by its dimensions for validation during loading.

## Limitations

- **Toy scale**: Designed for educational purposes, not production use
- **Single layer**: Only one attention + FFN block
- **Character-level**: Uses ASCII characters, not subword tokenization
- **No batching**: Processes sequences one at a time
- **Greedy sampling**: No temperature, top-k, or nucleus sampling
- **Fixed hyperparameters**: Requires recompilation to change model size

## Extending the Code

### Adding More Layers

Duplicate the attention and FFN blocks in the `Transformer` struct and update forward/backward passes:

```c
// Add second layer weights
Tensor w_q2, w_k2, w_v2, w_o2;
Tensor ln3_g, ln3_b, w3, b3, w4, b4, ln4_g, ln4_b;
```

### Custom Training Data

Replace the hardcoded string in `main()`:

```c
char* text = "Your custom training text here...";
```

### Implementing Adam Optimizer

Add momentum and adaptive learning rate tracking to `update_param()`:

```c
// First moment: m = beta1 * m + (1-beta1) * grad
// Second moment: v = beta2 * v + (1-beta2) * grad²
// Update: param -= lr * m / (sqrt(v) + epsilon)
```

### Batch Processing

Modify tensors to include a batch dimension and update matmul operations accordingly.

## Performance

On a modern CPU (Intel i7/M1), approximate performance:

- **Training step**: ~5-10ms per iteration
- **Full training (2000 steps)**: ~10-20 seconds
- **Inference**: <1ms per generated token

Performance can be improved with:
- SIMD vectorization (AVX, NEON)
- OpenMP parallelization
- CUDA/GPU acceleration
- Optimized BLAS libraries (OpenBLAS, MKL)

## Educational Value

This implementation demonstrates:

1. **Forward propagation**: How data flows through neural network layers
2. **Backpropagation**: Reverse-mode automatic differentiation by hand
3. **Attention mechanism**: The core innovation behind Transformers
4. **Gradient descent**: How neural networks learn from data
5. **Numerical stability**: Techniques like max subtraction in softmax
6. **Memory management**: Manual allocation and tracking in C

## Troubleshooting

### Compilation Errors

**Error**: `undefined reference to 'sqrt'`
- **Solution**: Add `-lm` flag to link math library

**Error**: `implicit declaration of function`
- **Solution**: Ensure using C99 or later (`-std=c99`)

### Runtime Issues

**NaN/Inf gradients**
- Increase gradient clipping threshold
- Reduce learning rate
- Check for numerical instability in softmax

**Segmentation fault**
- Verify `model.bin` exists for inference mode
- Check tensor dimensions match expected sizes

**Poor generation quality**
- Train for more iterations
- Use more diverse training data
- Increase model capacity (D_MODEL, D_FF)


---

**Project Status**: Educational/Experimental  
**Author**: aam-007  
**Version**: 1.0.0