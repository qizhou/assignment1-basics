# Train tinystories

from tokenizer import DATA_PATH, get_batch
from llm import AdamW, TransformerLM, calc_cross_entropy
import numpy as np
import torch

vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
rope_theta = 10000
num_layers = 4
num_heads = 16
datafile = DATA_PATH / "TinyStoriesV2-GPT4-train.txt.npy"

# total_tokens_processed = 327680000 # for GPU
total_tokens_processed = 40000000 # for CPU
batch_size = 32
steps = total_tokens_processed // batch_size // context_length
device = "cpu"

tokens = np.load(datafile, "r")
transformer = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
optimizer = AdamW(transformer.parameters(), lr=1)

for _ in range(steps):
    input_ids, target_ids = get_batch(tokens, batch_size, context_length, device)

    logits = transformer(input_ids)

    # reshape logics from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size]
    # reshape target_ids from [batch, seq_len] to [batch * seq_len]
    loss = calc_cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
    print(loss)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

