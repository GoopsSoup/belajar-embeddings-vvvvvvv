from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

basic = "I love unique weather"

sentences = [
    "I love rain",
    "I love today weather",
    "Today's weather doesn't seems to be that great",
    "What a lovely weather it is today"
    "I don't like rain",
    "I hate bad weather",
    "I like this type of weather it seems warm"
]

# embedding = model.encode(sentences)
# print(embedding.shape)
# similarities = model.similarity(embedding, embedding)
# print(similarities)

ranks = model.rank(basic, sentences)

print("Query", basic)
for rank in ranks:
    print(f"{rank["score"]:.2f}\t{sentences[rank['corpus_id']]}")
