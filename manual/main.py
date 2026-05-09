import numpy as np

words = ["We", "like", "love", "hate", "rain", "cats", "dawg", "snowing", "sunny"]
word_idx = {w:i for i, w in enumerate(words)}

embed = np.random.rand(len(words), 2)

def vector(w):
    return embed[word_idx[w]]

v_input = vector("love")
v_pos = vector("cats")
v_neg = vector("dawg")

for epoch in range(100):

    old_input = v_input.copy()
    old_pos = v_pos.copy()
    old_neg = v_neg.copy()

    v_input += 0.01 * old_pos
    v_pos += 0.01 * old_input

    v_input -= 0.01 * old_neg
    v_neg -= 0.01 * old_input

    score_pos = np.dot(v_input, v_pos)
    score_neg = np.dot(v_input, v_neg)

    print(score_pos, score_neg)