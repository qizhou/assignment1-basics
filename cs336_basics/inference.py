import pickle
import pathlib
from tokenizer import Tokenizer
import torch
from llm import TransformerLM, load_checkpoint


DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

prompt = "One day"

datafile = "TinyStoriesV2-GPT4-train.txt"
checkpoint_file = "tsv2_ttp327680000_step19999_batch128.chkpnt" # ~1.28

vocab_path = DATA_PATH / (datafile + ".vocab")
merges_path = DATA_PATH / (datafile + ".merge")

with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
with open(merges_path, "rb") as f:
    merges = pickle.load(f)

vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
rope_theta = 10000
num_layers = 4
num_heads = 16
batch_size = 64
device = "cuda"

# prompt to tokenids
tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
ids = tokenizer.encode(prompt)
if len(ids) > context_length:
    ids = ids[-context_length:]

# load model
model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=device)
model = torch.compile(model)
model.eval() # avoid dropout
checkpoint = load_checkpoint(checkpoint_file, model, optimizer=None)

# generate
pos = len(ids) # position to predict
ids.extend([0 for _ in range(context_length - len(ids))])
print(prompt, end="")
temperature = 0.5
top_p = 0.8
while True:
    batch = torch.tensor([ids[-context_length:]], device=device)
    logits = model(batch)
    if pos < context_length:
        data = logits[0, pos-1, :].reshape(-1)
    else:
        data = logits[0, context_length-1, :].reshape(-1)
    ex = (data / temperature).exp()
    ex_sum = ex.sum()
    prob = ex / ex_sum
    if top_p < 1:
        sorted, sorted_idx = torch.sort(prob, descending=True)
        cum_p = 0
        for i, x in enumerate(sorted):
            cum_p += x.item()
            if cum_p >= top_p:
                index = torch.multinomial(sorted[:i+1], 1)
                index = sorted_idx[index]
                break
    else:
        index = torch.multinomial(prob, 1)

    print(vocab[index.item()].decode("ascii"), end="")

    if pos >= len(ids):
        ids.append(index)
    else:
        ids[pos] = index
    pos += 1
    if vocab[index.item()] == b"<|endoftext|>":
        break
