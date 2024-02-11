import numpy as np
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer

class EmbeddingProcessing:
    def __init__(self, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self.word_tokenizer = None
        self.vocabulary_size = None
        self.embedding_model = None
        self.embeddings = None
        self.embeddings_weights = None

    def load_embedding_model(self, model_path):
        self.embedding_model = gensim.models.Word2Vec.load(model_path)

    def create_embeddings(self, word_tokenizer):
        self.word_tokenizer = word_tokenizer
        self.vocabulary_size = len(word_tokenizer.word_index) + 1

        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Use load_embedding_model method first.")

        # Initialize an empty dictionary to store word embeddings
        self.embeddings = {}

        # Populate the embeddings dictionary with words and their vectors
        for word in self.embedding_model.wv.index_to_key:
            vector = self.embedding_model.wv.get_vector(word)
            # Convert the vector to a numpy array of float32
            coefs = np.array(vector, dtype='float32')
            self.embeddings[word] = coefs

        # Initialize an empty matrix for word embeddings
        self.embeddings_weights = np.zeros((self.vocabulary_size, self.embedding_dim))

        # Populate the embeddings_weights matrix with vectors for words in the tokenizer's vocabulary
        for word, i in self.word_tokenizer.word_index.items():
            # Check if the word is in the embeddings dictionary
            embedding_vector = self.embeddings.get(word)

            if embedding_vector is not None:
                # If the word is present, assign its vector to the corresponding row in the matrix
                self.embeddings_weights[i] = embedding_vector

        return self.embeddings_weights