import regex as re
from tokenizer import merge_once

text = b"""low low low low low lower lower widest widest widest newest newest newest newest newest newest"""

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = bytes(PAT, "ascii")
table = {}
matches = re.finditer(PAT, text)
for w in matches:
    k = tuple([bytes([c]) for c in w.group()])
    table[k] = table.get(k, 0) + 1

# note that unlike the example in class note, the whitespace is included in split words.
print(table)

for i in range(6):
    merged_item, table = merge_once(table)
    print(merged_item[0], table)



