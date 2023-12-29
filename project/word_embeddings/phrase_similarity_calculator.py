import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

class PhraseSimilarityCalculator:
    def __init__(self, phrases_df, word2vec_loader):
        self.phrases_df = phrases_df
        self.word2vec_loader = word2vec_loader
        self.embeddings_matrix = np.vstack(self.phrases_df['Phrases'].apply(self._get_phrase_embedding).to_numpy())

    def calculate_distances(self, distance_metric='cosine'):
        if distance_metric == 'euclidean':
            distance_matrix = euclidean_distances(self.embeddings_matrix, self.embeddings_matrix)
        elif distance_metric == 'cosine':
            distance_matrix = cosine_distances(self.embeddings_matrix, self.embeddings_matrix)
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        return distance_matrix

    def find_closest_match(self, input_phrase, distance_metric='cosine'):
        input_embedding = self._get_phrase_embedding(input_phrase)
        if distance_metric == 'euclidean':
            distances = euclidean_distances([input_embedding], self.embeddings_matrix)[0]
        elif distance_metric == 'cosine':
            distances = cosine_distances([input_embedding], self.embeddings_matrix)[0]
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        closest_match_index = np.argmin(distances)
        closest_match = self.phrases_df.loc[closest_match_index, 'Phrases']
        return closest_match, distances[closest_match_index]

    def _get_phrase_embedding(self, phrase):
        words = phrase.split()
        embeddings = [self.word2vec_loader.get_word_embedding(word) for word in words]
        valid_embeddings = [emb for emb in embeddings if emb is not None]

        if not valid_embeddings:
            # If no valid embeddings found for any word, return a zero vector
            return np.zeros(self.word2vec_loader.model.vector_size)

        return np.mean(valid_embeddings, axis=0)