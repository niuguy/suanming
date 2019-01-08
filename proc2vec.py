import gensim 
import logging
import pandas as pd
import _pickle as pickle
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from dataset import dtoc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import mlflow
from mlflow import log_metric
import numpy as np
import time 
import argparse
from sklearn.utils import shuffle



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
def train_word2vec(dataset, model_type=0, size = 150):
    model = Word2Vec(dataset,size=size,window=3,min_count=1,workers=10,sg=model_type)
    model.train(dataset, total_examples=len(dataset), epochs=10)
    return model

def generate_training_dataset(filename = 'dataset/dtoc_proc.pkl'): 
    pd_dtoc = pickle.load(open(filename, 'rb'))
    pd_values = pd_dtoc[['proc1','proc2','proc3','proc4','proc5','proc6', 
    'proc7', 'proc8', 'proc9','proc10', 'proc11', 'proc12']].values
    rst = []
    for a in pd_values:
        e = [x for x in a if x is not None]
        rst.append(e)
    return rst

def train(model_type, dim):
    start_time = time.time()
    with mlflow.start_run():
        dataset = generate_training_dataset(filename = 'dataset/dtoc_proc.pkl')
        artifact_name = "models/proc2vec_" + str(model_type) + "_" + str(dim) + ".pkl"
        model = train_word2vec(dataset, model_type, size=dim)
        model.save(artifact_name)
        mlflow.log_param("model_type", model_type)
        mlflow.log_artifact(artifact_name)
    timed = time.time() - start_time
    print("This model took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="Choose from CBOW(0),Skip-Gram(1)", action = 'store', default=1, type = str)
    parser.add_argument("--dim", help="Embedding dimension", nargs='?', action='store', default=200, type=float)
    
    args = parser.parse_args()
    model_type = args.model_type
    dim = args.dim

    print('Model type:', model_type)
    print('Embedding dimension:', dim)
    
    train(model_type, dim)