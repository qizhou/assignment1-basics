# Train tinystories

from tokenizer import DATA_PATH, get_batch
from llm import AdamW, TransformerLM, calc_cross_entropy, get_lr_cosine_schedule, clip_gradient, save_checkpoint
import numpy as np
import torch

vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
rope_theta = 10000
num_layers = 4
num_heads = 16
name = "tsv2"
datafile = DATA_PATH / "TinyStoriesV2-GPT4-train.txt.npy"

total_tokens_processed = 327680000 # for GPU
# total_tokens_processed = 40000000 # for CPU
batch_size = 64
device = "cuda"

# learning scheduling
max_lr = 3e-3
min_lr = max_lr / 10          # 10% of max
warmup_steps = 500
post_annealing_steps = 0

# checkpoint
steps_per_checkpoint = 1000

if device != "mps":
    torch.set_float32_matmul_precision('high') # not for mps

def train(vocab_size, context_length, d_model, d_ff, rope_theta, num_layers, num_heads, name, datafile, total_tokens_processed, batch_size, device, max_lr, min_lr, warmup_steps, post_annealing_steps, steps_per_checkpoint):
    tokens = np.load(datafile)
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device)
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=max_lr)

    max_steps = total_tokens_processed // batch_size // context_length

    for step in range(max_steps):
        input_ids, target_ids = get_batch(tokens, batch_size, context_length, device)

        logits = model(input_ids)

    # reshape logics from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size]
    # reshape target_ids from [batch, seq_len] to [batch * seq_len]
        loss = calc_cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        lr = get_lr_cosine_schedule(step+1, max_lr, min_lr, warmup_steps, max_steps-post_annealing_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        print(step, loss, lr)

        loss.backward()

        clip_gradient(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % steps_per_checkpoint == 0:
            print("saving checkpoint")
            save_checkpoint(model, optimizer, step, f"{name}_ttp{total_tokens_processed}_batch{batch_size}_step{step}.chkpnt")

    save_checkpoint(model, optimizer, step, f"{name}_ttp{total_tokens_processed}_batch{batch_size}_step{step}.chkpnt")


for batch_size in [32, 64, 128, 256]:
    train(vocab_size, context_length, d_model, d_ff, rope_theta, num_layers, num_heads, name, datafile, total_tokens_processed, batch_size, device, max_lr, min_lr, warmup_steps, post_annealing_steps, steps_per_checkpoint)
