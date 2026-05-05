from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np


model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

basic = "I love unique weather"

sentences = [
    "I love rain",
    "I love today weather",
    "Today's weather doesn't seems to be that great",
    "What a lovely weather it is today",
    "I don't like rain",
    "I hate bad weather",
    "I like this type of weather it seems warm"
] 

#comparison v
# ranks = model.rank(basic)

# print("Query    ", ranks)
# for rank in ranks:
#     print(f"{rank["score"]:.2f}\t{sentences[rank['corpus_id']]}")
     
# berbeda
    
sentences_combinations = [[basic, sentence] for sentence in sentences]
score = model.predict(sentences_combinations)

ranked_indices = np.argsort(score)[::-1]
print("Score", score)
print("Query", ranked_indices)

