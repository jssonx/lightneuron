# Embeddings

## How to train and use the word2vec model
1. Download and preprocess data: `python3 get_data.py`
2. Compile code: `gcc -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result word2vec.c -o word2vec -lm`
3. Train the model: `./word2vec -train brown.txt -output vectors.bin -cbow 1 -size 200 -window 5 -negative 5 -hs 0 -sample 1e-3 -threads 12 -binary 1 -iter 15`
4. Use the model: `python3 word2vec.py`