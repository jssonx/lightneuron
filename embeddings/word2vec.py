from gensim.models import KeyedVectors

# Load pre-trained word2vec model
model = KeyedVectors.load_word2vec_format('vectors.bin', binary=True)

# Find the 10 words most similar to 'woman'
similar_words = model.most_similar('woman', topn=10)
for word, similarity in similar_words:
    print(word, similarity)
