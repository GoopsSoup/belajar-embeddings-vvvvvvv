import numpy as np

words = ["hi", "hello", "fine", "llove"]
word = {}

for i, w in enumerate(words):
    word[w] = i
    
print(word['fine'])