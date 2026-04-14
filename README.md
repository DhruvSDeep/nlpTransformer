# Natural Language Transformer

A GPT-style decoder-only transformer language model built from scratch in PyTorch. Includes a custom Byte Pair Encoding (BPE) tokenizer and autoregressive text generation with temperature and top-k sampling.

---

## Highlights

- **From-scratch transformer** — multi-head causal self-attention, sinusoidal positional encoding, pre-norm residual blocks, and SwiGLU-style feed-forward layers.
- **Custom BPE tokenizer** — learns subword merges directly from your corpus; no external tokenizer library required.
- **Weight tying** — the token embedding matrix is shared with the output projection head for better parameter efficiency.
- **Modern training recipe** — AdamW with weight decay, cosine learning rate schedule with linear warm-up, mixed-precision (AMP + GradScaler), and gradient clipping.
- **Top-k sampling with temperature** — controllable text generation at inference time.

---

## Architecture

| Component | Detail |
|---|---|
| Type | Decoder-only (GPT-style) |
| Embedding dimension | 256 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Feed-forward dimension | 1024 |
| Context length | 512 tokens |
| Vocabulary size | ~3 000 BPE tokens |
| Feed-forward activation | SiLU gating (SwiGLU variant) |
| Normalization | Pre-LayerNorm |
| Dropout | 0.2 (attention), 0.1 (embeddings) |

---

## Project Structure

```
nlpTransformer/
├── src/
│   ├── main.py               # Training entry point — data loading, tokenization, training loop
│   ├── transformerLogic.py    # Model architecture, training function, generation function
│   ├── dataLogic.py           # BPE tokenizer, word frequency counting, PyTorch Dataset
│   ├── generate.py            # Inference script — load weights & generate text from a prompt
│   └── tests.py               # Utility / debugging scripts
├── data/
│   ├── frequencyDict.pkl      # Word frequency dictionary built from corpus
│   ├── manual.pkl             # Ordered BPE merge rules
│   ├── vocab.pkl              # Final vocabulary set
│   ├── token_to_idx.pkl       # Token-to-index mapping
│   ├── tokenizedFiles.pkl     # Pre-tokenized training data
│   └── valTokenizedFiles.pkl  # Pre-tokenized validation data
├── checkpoints/               # Saved model weights (.pt files)
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.8+ (Use version according to CUDA and PyTorch specifications)
- PyTorch (with CUDA recommended for training)

```bash
pip install torch
```

### Prepare Your Data

Place `.txt` files anywhere under the `data/` directory (subdirectories are searched recursively). The pipeline will automatically build a BPE vocabulary and tokenize the corpus on the first run.

### Train

```bash
cd src
python main.py
```

On the first run, the script will:
1. Build a word frequency dictionary from all `.txt` files in `data/`.
2. Learn ~3 000 BPE merge rules.
3. Tokenize the corpus and create train/validation splits (95% / 5%).
4. Train for 80 epochs, saving the best checkpoint (by validation loss) to `checkpoints/`.

Training hyperparameters (configured in `transformerLogic.py`):

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Learning rate | 6 × 10⁻⁴ |
| Weight decay | 0.1 |
| Warm-up steps | 2 000 |
| LR schedule | Cosine annealing (after warm-up) |
| Gradient clipping | Max norm 1.0 |
| Epochs | 80 |

### Generate Text

```bash
cd src
python generate.py
```

Edit the `prompt` variable in `generate.py` to set your seed text. Generation parameters (temperature, top-k) can be adjusted in the same file.

---

## How It Works

### Tokenization (BPE)

1. **Word frequencies** — Every `.txt` file is scanned and a word-level frequency dictionary is built (`dataLogic.wordFreqDict`).
2. **Byte Pair Encoding** — Starting from individual characters, the most frequent adjacent pair is merged iteratively until the vocabulary reaches ~3 000 tokens (`dataLogic.bytePairEncode`).
3. **Encoding text** — At inference time, input text is split into words, then each word is greedily merged using the learned merge table in priority order (`dataLogic.tokenise`).

### Training

- Sequences of 512 tokens are created with a stride equal to the sequence length (non-overlapping chunks).
- The model is trained with **next-token prediction**: given tokens `[0 … N-1]`, predict `[1 … N]`.
- Cross-entropy loss ignores the `<PAD>` token (index 0).
- Mixed-precision training (`torch.cuda.amp`) is used for faster GPU throughput.
- The best model checkpoint is saved whenever validation loss improves.

### Generation

The `creation` function performs **autoregressive sampling**:
1. Feed the seed tokens through the model.
2. Apply temperature scaling to the final logits.
3. Zero out all logits below the top-k threshold.
4. Sample the next token from the resulting probability distribution.
5. Append and repeat until the `<EOS>` token is produced or the maximum length is reached.

---

## Checkpoints

The `checkpoints/` directory contains several pre-trained weight files at different training stages. To load a specific checkpoint for generation, update the path in `generate.py`:

```python
model.load_state_dict(torch.load("./checkpoints/<checkpoint_file>.pt"))
```

---

## License

No license specified. Contact the repository owner for usage terms.
