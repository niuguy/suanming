import time
import pickle
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random 

if __name__ == "__main__":
    horiz_model = Word2Vec.load("models/diag2vec_1_150.pkl")
    random.seed(0)
    print(random.sample(horiz_model.wv.index2word, 200))