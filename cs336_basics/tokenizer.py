PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import regex as re

# Merge
def merge(table: dict[tuple[str], int] | dict[tuple[bytes], int]):
    # counter the pair with the higest frequency
    counter = {}
    for k,v in table.items():
        for i in range(len(k)-1):
            counter[k[i:i+2]] = counter.get(k[i:i+2], 0) + v

    largest_item = max(counter.items(), key=lambda item: (item[1], item[0]))

    # merge
    # Q: suppose we have 'w','w','w', in counter, the pair 'w','w' has 2, but only merge once?
    new_table = {}
    replaced = largest_item[0][0] + largest_item[0][1]
    for k,v in table.items():
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