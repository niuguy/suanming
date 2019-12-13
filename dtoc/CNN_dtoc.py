import os
import time
import numpy as np # linear algebra
from numpy import zeros
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from gensim.models import Word2Vec

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate,MaxPooling2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import L1L2
from keras.utils import normalize
from sklearn import preprocessing

from dataset import dtoc
from keras.callbacks import TensorBoard
import mlflow


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def get_unique_codes(df):
    all_codes = set()
    for dig_name in df.columns[4:16]:
        for element in df[dig_name]:
            all_codes.add(element)
    return all_codes

def get_embeddings_matrix(df):
    EMBEDDING_MODEL_FILE = 'diag2vec.model'    
    diag2vec_model = Word2Vec.load(EMBEDDING_MODEL_FILE)

    embedding_index = diag2vec_model.wv
    print('Loaded %s code vectors.' % len(embedding_index))

    unique_codes = get_unique_codes(df)
    vocab_size = len(unique_codes)
    embedding_matrix = zeros((vocab_size, 150))
    for code in unique_codes:
        embedding_vector = embedding_index[code]
        if embedding_vector is not None:
            embedding_matrix[0] = embedding_vector
    return embedding_matrix 

def age_vectorization(df):
    min_max_scaler = preprocessing.MinMaxScaler() 
    vector_age = min_max_scaler.fit_transform(df[['age']].values.astype(float))

    vector_age = np.reshape(vector_age, (len(df['age']),1))
    vector_age = np.pad(vector_age, ((0,0),(0,149)), 'mean')
    return vector_age

def normalization(array):
    min_max_scaler = preprocessing.MinMaxScaler() 
    normal = min_max_scaler.fit_transform(array)
    return normal


def vectorization(df, wv_model):
    
    empty_vec = np.zeros(150)
    embeds = []
    for index, row in df.iterrows():
        row_embeds = []
        ## add age
        age = np.empty(150)
        age.fill(row['age']/100)       
        row_embeds.append(age)
        for dig in df[['diag1','diag2','diag3','diag4','diag5','diag6', 
    'diag7', 'diag8', 'diag9','diag10', 'diag11', 'diag12']].columns:
            if row[dig]:
                row_embeds.append(wv_model.wv[row[dig]])
            else:
                row_embeds.append(empty_vec)        
        embeds.append(row_embeds)
    return np.array(embeds)

def model_LSTM():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model

def model_CNN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(36800, 12, 150)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model

def model_LR():

    model = Sequential()
    model.add(Flatten())
    # model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(12, 150)))
    # model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1,  # output dim is 2, one score per each class
                kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                activation='relu',
                input_dim=3)) # input dimension = number of features your data h

                
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def train():
    ## load data

    df = dtoc.sample(40000)
    train_df, val_df = train_test_split(df, test_size=0.08, random_state=2018)

    train_X = train_df[['diag1','diag2','diag3','diag4','diag5','diag6', 
    'diag7', 'diag8', 'diag9','diag10', 'diag11', 'diag12', 'age']]

    # train_X = train_df['age'].values
    train_y = train_df['is_dtoc'].values
    
    val_X = val_df[['diag1','diag2','diag3','diag4','diag5','diag6', 
    'diag7', 'diag8', 'diag9','diag10', 'diag11', 'diag12', 'age']]
    # val_X = train_df['age'].values
    val_y = val_df['is_dtoc'].values

    # embed_size = 150 # how big is each variable vector
    # max_features = 10000 # how many unique codes to use (ICD codes + all ages)
    # max_len = 13 # max number of variables in one records
    # embedding_matrix = get_embeddings_matrix(df)
    # load word2vec model
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    EMBEDDING_MODEL_FILE = 'diag2vec.model'    
    wv_model = Word2Vec.load(EMBEDDING_MODEL_FILE)
    
    #model = model_LR()
    model = model_LSTM()
    # model = model_CNN()

    model.fit(vectorization(train_X, wv_model), train_y, batch_size=512, epochs=20, validation_data=(vectorization(val_X, wv_model), val_y),
    callbacks=[tbCallBack])

    
    pred_val_y = model.predict(vectorization(val_X, wv_model), batch_size=1024, verbose=1)
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))


if __name__ == "__main__":
    train()
    # df = dtoc.sample(4000)[:1]
    # print(vectorization(df, Word2Vec.load('diag2vec.model')))

    # # Word2Vec.load('diag2vec.model')

    # min_max_scaler = preprocessing.MinMaxScaler() 
    # vector_age = min_max_scaler.fit_transform(df[['age']].values.astype(float))

    # vector_age = np.reshape(vector_age, (len(df['age']),1))
    # vector_age = np.pad(vector_age, ((0,0),(0,9)), 'mean')

    # # vector_age_1 = vector_age[:1]
    # np.concatenate((np.zeros((1, 10)), vector_age), axis = 0)
    


   






    

