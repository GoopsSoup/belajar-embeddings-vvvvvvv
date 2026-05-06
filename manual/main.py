import numpy as np

words = ["We", "like", "love", "hate", "rain", "cats", "dawg", "snowing", "sunny"]
word_idx = {w:i for i, w in enumerate(words)}

embed = np.random.rand(len(words), 2)

def vector(w):
    return embed[word_idx[w]]

v_input = vector("love")
v_pos = vector("cats")
v_neg = vector("dawg")

lr = 0.01

print(lr * v_pos)
print(lr * v_neg)