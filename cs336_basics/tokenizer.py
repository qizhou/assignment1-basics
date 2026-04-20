import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import os
from typing import BinaryIO

from collections.abc import Iterable, Iterator

import json

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

                matches = re.finditer(bytes(PAT, "utf-8"), d)
                for w in matches:
                    # split and merge the tokens
                    ts = []
                    for c in w.group():
                        ts.append(bytes([c]))

                    # apply merges one by one
                    for m in self.merges:
                        i = 0
                        while i<len(ts)-1:
                            if m == (ts[i], ts[i+1]):
                                ts[i] = ts[i]+ts[i+1]
                                del ts[i+1]
                            else:
                                i += 1

                    for t in ts:
                        yield self.rev_vocab[t]

    def decode(self, ids: list[int]) -> str:
        bs = bytearray()
        for id in ids:
            bs.extend(self.vocab[id])
        return bs.decode("utf-8", errors="replace")
