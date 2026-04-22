import regex as re
PAT = bytes(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", "ascii")

import os
from typing import BinaryIO

from collections.abc import Iterable, Iterator

import json
import pickle

from multiprocessing import Process, Queue
import pathlib
import numpy
import numpy.typing as npt
import torch
from random import randrange


DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def merge_once(
    table: dict[tuple[str, ...], int] | dict[tuple[bytes, ...], int]
) -> tuple[tuple[tuple[bytes, bytes], int], dict[tuple[str, ...], int] | dict[tuple[bytes, ...]]]:
    """
    A slow version of merge.
    Args:
        table (dict[tuple[str, ...], int] | dict[tuple[bytes, ...], int]): map of words that are splitted in tokens to frequency count

    Returns:
        tuple[tuple[tuple[bytes, bytes], int], dict[tuple[str, ...], int] | dict[tuple[bytes, ...]]]:
            merged_token: tokens to be merged with frequency
            new_table: the new table with merged tokens
    """

    # counter the pair with the higest frequency
    counter = {}
    for k, v in table.items():
        for i in range(len(k)-1):
            counter[k[i:i+2]] = counter.get(k[i:i+2], 0) + v

    largest_item = max(counter.items(), key=lambda item: (item[1], item[0]))

    # merge
    # Q: suppose we have 'w','w','w', in counter, the pair 'w','w' has 2, but only merge once?
    new_table = {}
    replaced = largest_item[0][0] + largest_item[0][1]
    for k, v in table.items():
        l = []
        i = 0
        while i < len(k):
            if i < len(k)+1 and k[i:i+2] == largest_item[0]:
                l.append(replaced)
                i = i + 1
            else:
                l.append(k[i])
            i = i + 1
        new_table[tuple(l)] = v
    return largest_item, new_table


def merge(
    table: dict[tuple[str, ...], int] | dict[tuple[bytes, ...], int],
    times: int
):
    # counter the pair with the higest frequency
    counter = {}
    rev_counter = {}

    # convert table to list
    table = [(k, v) for k, v in table.items()]
    for j in range(len(table)):
        k, v = table[j]
        for i in range(len(k)-1):
            counter[k[i:i+2]] = counter.get(k[i:i+2], 0) + v
            if k[i:i+2] not in rev_counter:
                rev_counter[k[i:i+2]] = set()
            rev_counter[k[i:i+2]].add(j)

    merges = []
    for _ in range(times):
        largest_item = max(counter.items(), key=lambda item: (item[1], item[0]))
        merges.append(largest_item[0])

        idx = rev_counter[largest_item[0]]

        # merge
        # Q: suppose we have 'w','w','w', in counter, the pair 'w','w' has 2, but only merge once?
        replaced = largest_item[0][0] + largest_item[0][1]
        for j in set(idx):
            k, v = table[j]
            i = 0
            l = []
            while i < len(k):
                if i < len(k)+1 and k[i:i+2] == largest_item[0]:
                    l.append(replaced)
                    i = i + 1
                else:
                    l.append(k[i])
                i = i + 1
            new_k = tuple(l)

            # update the table entry
            table[j] = (new_k, v)

            # update counters
            for i in range(len(k)-1):
                counter[k[i:i+2]] -= v
                rev_counter[k[i:i+2]].discard(j)  # may remove twice
            for i in range(len(new_k)-1):
                counter[new_k[i:i+2]] = counter.get(new_k[i:i+2], 0) + v
                if new_k[i:i+2] not in rev_counter:
                    rev_counter[new_k[i:i+2]] = set()
                rev_counter[new_k[i:i+2]].add(j)

    return merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.rev_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.rev_merges = {merges[i]: i for i in range(len(merges))}
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = sorted(special_tokens, key=lambda x: (-len(x),x)) # sort the tokens so that we will match longest first
        self.special_tokens_map = {bytes(t, "utf-8"): self.rev_vocab[bytes(t, "utf-8")] for t in self.special_tokens}
        self.special_tokens_escaped = [re.escape(bytes(t, "utf-8")) for t in self.special_tokens]

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            merges = json.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return [x for x in self.encode_iterable([text])]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            bs = s.encode("utf-8")

            docs = [bs]
            if len(self.special_tokens) != 0:
                # check special tokens
                docs = re.split(b"("+b"|".join((self.special_tokens_escaped))+b")", bs)

            for d in docs:
                # if it is special token, return directly
                if d in self.special_tokens_map:
                    yield self.special_tokens_map[d]
                    continue

                matches = re.finditer(PAT, d)
                for w in matches:
                    # split and merge the tokens
                    ts = []
                    for c in w.group():
                        ts.append(bytes([c]))

                    # apply merges one by one
                    # slow version
                    # for m in self.merges:
                    #     i = 0
                    #     while i<len(ts)-1:
                    #         if m == (ts[i], ts[i+1]):
                    #             ts[i] = ts[i]+ts[i+1]
                    #             del ts[i+1]
                    #         else:
                    #             i += 1
                    # faster version: reduce the time from 12s to 3s
                    while True:
                        min_merge = None
                        for i in range(len(ts)-1):
                            merge_key = (ts[i], ts[i+1])
                            merge_idx = self.rev_merges.get(merge_key, None)
                            if merge_idx is not None and (min_merge is None or merge_idx < min_merge):
                                min_merge = merge_idx
                                min_merge_pos = i
                        if min_merge is None:
                            break
                        ts[min_merge_pos] = self.merges[min_merge][0] + self.merges[min_merge][1]
                        del ts[min_merge_pos+1]

                    for t in ts:
                        yield self.rev_vocab[t]

    def decode(self, ids: list[int]) -> str:
        bs = bytearray()
        for id in ids:
            bs.extend(self.vocab[id])
        return bs.decode("utf-8", errors="replace")


def convert(q, s, special_tokens):
    # convert docs into table as word => freq.
    docs = re.split(bytes("|".join(special_tokens), "ascii"), s)

    table = {}
    for d in docs:
        matches = re.finditer(PAT, d)
        for w in matches:
            k = tuple([bytes([c]) for c in w.group()])
            table[k] = table.get(k, 0) + 1
    q.put(table)


def train_tokenizer(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    nproc: int = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # read all data and split into words
    # with open(input_path, "rb") as f:
    #     s = f.read() # into bytes

    # the init stage of multiprocessing on mac is quite slow:
    # - for test_train_bpe_speed, the time is increased from 0.18s to 1.03s
    p_list = []
    q = Queue()
    with open(input_path, "rb") as f:
        num_processes = nproc if nproc is not None else 6
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start)
            p_list.append(Process(target=convert, args=(q, chunk, special_tokens)))
            p_list[-1].start()
        # Run pre-tokenization on your chunk and store the counts for each pre-token

    # merge all sub tables
    table = {}
    for p in p_list:
        sub_table = q.get()
        for k, v in sub_table.items():
            table[k] = table.get(k, 0) + v

    for p in p_list:
        p.join()


    # add all ascii
    vocab = {i: bytes([i]) for i in range(256)}

    # add special tokens
    vocab.update({i+256: bytes(special_tokens[i], "ascii") for i in range(len(special_tokens))})

    # merge pair-wise tokens with highest frequency
    # merges = []
    # while len(vocab) < vocab_size:
    #     merged_token, table = merge_once(table)
    #     merges.append(merged_token[0])
    #     vocab[len(vocab)] = merged_token[0][0] + merged_token[0][1]
    # in test_train_bpe_speed, this reduces the time from 1.76s to 0.18s
    merges = merge(table, vocab_size-len(vocab))
    for m in merges:
        vocab[len(vocab)] = m[0] + m[1]
    return vocab, merges


def encode_file(datafile, inputfile=None):
    inputfile = DATA_PATH / inputfile if inputfile is not None else DATA_PATH / datafile
    vocab_path = DATA_PATH / (datafile + ".vocab")
    merges_path = DATA_PATH / (datafile + ".merge")
    output_path = DATA_PATH / (datafile + ".npy")

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    with open(inputfile, "r") as f:
        data = f.read()
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    ids = tokenizer.encode(data)
    arr = numpy.array(ids, dtype=numpy.uint16)
    numpy.save(output_path, arr)


def tokenize_file(datafile):
    input_path = DATA_PATH / datafile
    vocab, merges = train_tokenizer(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    vocab_path = DATA_PATH / (datafile + ".vocab")
    merges_path = DATA_PATH / (datafile + ".merge")

    with open(vocab_path, "wb") as vocab_f:
        pickle.dump(vocab, vocab_f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    pos = [randrange(0, dataset.size - context_length) for _ in range(batch_size)]
    batch = torch.tensor([dataset[st:st+context_length] for st in pos], device=device)
    targets = torch.tensor([dataset[st+1:st+1+context_length] for st in pos], device=device)
    return (batch, targets)


if __name__ == "__main__":
    # train tokenizer based on TinyStores and OpenWebText
    datafile = "TinyStoriesV2-GPT4-train.txt"
    # datafile = "owt_train.txt"
    # tokenize_file(datafile)
    # encode_file(datafile, inputfile="TinyStoriesV2-GPT4-valid.txt")
    encode_file(datafile)