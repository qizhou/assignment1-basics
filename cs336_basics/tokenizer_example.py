from tokenizer import merge

text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

freqs = {'low': 5, 'lower': 2, 'widest': 3, 'newest': 6}

table = {tuple(k): v for k,v in freqs.items()}

for i in range(6):
    merged_item, table = merge(table)
    print(merged_item[0], table)



