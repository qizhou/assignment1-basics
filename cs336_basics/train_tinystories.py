# Train tinystories

from tokenizer import DATA_PATH, get_batch
from llm import AdamW, TransformerLM, calc_cross_entropy, get_lr_cosine_schedule, clip_gradient
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
max_steps = total_tokens_processed // batch_size // context_length
device = "cuda"

max_lr = 3e-3
min_lr = 3e-4          # 10% of max
warmup_steps = 500
post_annealing_steps = 0

if device != "mps":
    torch.set_float32_matmul_precision('high') # not for mps
tokens = np.load(datafile)
transformer = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device)
transformer = torch.compile(transformer)
optimizer = AdamW(transformer.parameters(), lr=1)

for step in range(max_steps):
    input_ids, target_ids = get_batch(tokens, batch_size, context_length, device)

    logits = transformer(input_ids)

    # reshape logics from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size]
    # reshape target_ids from [batch, seq_len] to [batch * seq_len]
    loss = calc_cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

    lr = get_lr_cosine_schedule(step+1, max_lr, min_lr, warmup_steps, max_steps-post_annealing_steps)
    for g in optimizer.param_groups:
        g["lr"] = lr

    print(step, loss, lr)

    loss.backward()

    clip_gradient(transformer.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

