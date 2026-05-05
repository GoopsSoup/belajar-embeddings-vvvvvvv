from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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


embedding = model.encode(sentences)
print(embedding.shape)
similarities = model.similarity(embedding, embedding)
print(similarities)


# permulaan
# embedding adalah teknik untuk mengubah sebuah kata atau kalimat
# menjadi deretan angka yang dapat dipahami komputer 
# dengan embedding kita dapat membedakan 2 kata berbeda menggunkan komputer 
# sebagai contoh ("kucing" dan "anak kucing") kedua itu adalah kalimat berbeda
# tapi akan diletakkan berdekatan karena mirip ketika sudah di embedding 
# 
# untuk cara bekerjanya kurang tau