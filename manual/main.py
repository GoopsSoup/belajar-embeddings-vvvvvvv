import numpy as np

words = ["I", "love", "cat", "and", "rain"]
word_idx = {w:i for i, w in enumerate(words)}

embed = np.random.rand(5,2)

def vector(wrd):
    return embed[word_idx[wrd]]
    
def dot(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similar = vector("love")

for w in words:
    score = dot(similar, vector(w))
    print(w, score)
