from cs336_basics.tokenizer import pretokenize, merge_pair  

result = pretokenize("Hello, I don't have 42 cats!")
print(result)
#[b'Hello', b',', b' I', b' don', b"'t", b' have', b' 42', b' cats', b'!']

# merge_pair
result = merge_pair((72, 101, 108, 108, 111), (108, 108), 256)
print(result)  # (72, 101, 256, 111)

# continue pair 
result = merge_pair((108, 108, 108), (108, 108), 256)
print(result)  # (256, 108)