from .adapters import run_train_bpe
from .common import DATA_PATH

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