from tokenizer import DATA_PATH
import numpy as np
from llm import TransformerLM, calc_cross_entropy, load_checkpoint
import torch

checkpoint_file = "tsv2_ttp40000000_step2440_batch64.chkpnt" # ~1.62
checkpoint_file = "tsv2_ttp327680000_step19999_batch128.chkpnt" # ~1.28
checkpoint_file = "tsv2_ttp327680000_step19999_batch64.chkpnt" # ~1.35
checkpoint_file = "tsv2_ttp327680000_step19999_batch32.chkpnt" # ~1.45
datafile = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt.npy" #

vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
rope_theta = 10000
num_layers = 4
num_heads = 16
batch_size = 64

tokens = np.load(datafile)

device = "cuda"

if device != "mps":
    torch.set_float32_matmul_precision('high') # not for mps

model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device)
model = torch.compile(model)
checkpoint = load_checkpoint(checkpoint_file, model, optimizer=None)

dataset = np.load(datafile)
total_loss = 0
with torch.no_grad():
    for pos_start in range(0, len(tokens) - context_length - 1, batch_size):
        batch_size = min(len(tokens)- context_length - pos_start - 1, batch_size)
        pos = range(pos_start, pos_start+batch_size)
        batch = torch.tensor([dataset[st:st+context_length] for st in pos], device=device, dtype=torch.int)
        targets = torch.tensor([dataset[st+1:st+1+context_length] for st in pos], device=device, dtype=torch.int)
        logits = model(batch)
        loss = calc_cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss * batch_size
        print(pos_start, loss, total_loss / (pos_start + batch_size))
    print("total avg loss", total_loss / (len(tokens) - context_length - 1))