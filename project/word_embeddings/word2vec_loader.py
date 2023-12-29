import gensim

class Word2VecLoader:
    def __init__(self,file_path,limit=1000000):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True,limit=limit)
    def get_word_embedding(self, word):
        return self.model[word] if word in self.model else None