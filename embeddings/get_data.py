import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import re

# Download the Brown corpus
nltk.download('brown')

# Get all sentences from the Brown corpus
sentences = brown.sents()

# Define a function to clean text
def clean_text(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove non-letter characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create a new list to store cleaned sentences
clean_sentences = []

# Iterate through each sentence in the Brown corpus
for sentence in sentences:
    # Convert the sentence from a list to a string
    sentence = ' '.join(sentence)
    # Clean the sentence
    sentence = clean_text(sentence)
    # Add the cleaned sentence to the new list
    clean_sentences.append(sentence)

# Write cleaned sentences to a new text file
with open('brown.txt', 'w') as f:
    for sentence in clean_sentences:
        f.write("%s\n" % sentence)
