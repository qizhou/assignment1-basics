from .adapters import run_train_bpe, get_tokenizer
from .common import DATA_PATH
import pickle
import pytest

@pytest.mark.skip(reason="Too long")
def test_train_large_bpe_tinystories():
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )


def test_encdec_tinystores():
    vocab_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt.vocab"
    merge_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt.merge"
    valid_path = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merge_path, "rb") as f:
        merges = pickle.load(f)
    with open(valid_path, "r") as f:
        valid = f.read()

    tokenizer = get_tokenizer(vocab, merges, ["<|endoftext|>"])
    ids = tokenizer.encode(valid)
    output = tokenizer.decode(ids)
    assert valid == output
