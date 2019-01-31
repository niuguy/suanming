# preprocess the input variables ,make it ready to feed into the prediction model

import numpy as np
import time 
import math
import argparse
import _pickle as pickle
from gensim.models import Word2Vec
from dataset import dtoc
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from models import embedding
from dataset import dtoc
import keras
import _pickle as pickle


if __name__ == "__main__":
    # df_in ->columns=['diag_hist1','diag_hist2','diag_hist3','diag_hist4','diag_hist5','diag2','diag3','diag4','diag5','diag6', 
    # 'diag7', 'diag8', 'diag9','diag10', 'diag11', 'diag12', 'age', 'adm_code', 'gender','adm_month','is_oversea','dest_code'])

    df_in = pickle.load(open('predict_case.pkl', 'rb'))
    print('df_in--------')
    print(df_in)
    
    emb_file_diags = "models/diag2vec_1_150.pkl"
    emb_file_seqs = "models/seq2vec_1_150.pkl"
    wv_model_seqs = Word2Vec.load(emb_file_seqs)
    wv_model_diags = Word2Vec.load(emb_file_diags)

    df_X = embedding.embed_X(df_in, wv_model_diags,wv_model_seqs)
    print('df_X---------')
    print(df_X)
    saved_model_dir = 'PlanetModel/deep_pred_model.h5' 
    pred_model = keras.models.load_model(saved_model_dir)
    pred_Y = pred_model.predict(df_X)
    print('pred_Y', pred_Y)


    


    