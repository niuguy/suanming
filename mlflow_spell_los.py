import os
import time
import argparse
import numpy as np 
from numpy import zeros
import pandas as pd 
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from gensim.models import Word2Vec

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv1D, MaxPool1D, concatenate,MaxPooling2D
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
from dataset import dtoc,los
from keras.callbacks import TensorBoard
import mlflow
from mlflow import log_metric
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def run_LR(train_X, train_y, eps, bsize):
    # define base model
    def baseline_model():
        # create model
        model = Sequential()
        # model.add(Dense(5700, input_dim=5700, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=baseline_model, epochs=eps, batch_size=bsize, verbose=0)

    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, train_X, train_y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    return results.mean(), results.std()
    


def run_model(model, train_X, train_y, val_X, val_y, bsize, eps):

    model.fit(train_X, train_y, batch_size=bsize, epochs=eps, validation_data = (val_X, val_y))

    predict_val_y = model.predict(val_X, batch_size=bsize, verbose=1)
    predict_val_y = (predict_val_y>0.5).astype(int)
    acc_score = accuracy_score(val_y, predict_val_y)
    f1_score = metrics.f1_score(val_y, predict_val_y)
    roc_auc_score = metrics.roc_auc_score(val_y, predict_val_y)
    print("Classification report:\n%s\n"
      % (metrics.classification_report(val_y, predict_val_y)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(val_y, predict_val_y))

    return acc_score, f1_score, roc_auc_score
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="Choose from LSTM,LSTM_ATTENTION, CNN, LR", action = 'store', default="LR", type = str)
    parser.add_argument("--drop_rate", help="Drop rate", nargs='?', action='store', default=0.5, type=float)
    parser.add_argument("--input_dim", help="Input dimension for the network.", action='store', nargs='?', default=20, type=int)
    parser.add_argument("--bs", help="Number of rows or size of the tensor", action='store', nargs='?', default=1000, type=int)
    parser.add_argument("--output", help="Output from First & Hidden Layers", action='store',  nargs='?', default=64, type=int)
    parser.add_argument("--train_batch_size", help="Training Batch Size", nargs='?', action='store', default=1024, type=int)
    parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20, type=int)
    
    args = parser.parse_args()

    drop_rate = args.drop_rate
    input_dim = args.input_dim
    bs = args.bs
    output = args.output
    epochs = args.epochs
    batch_size = args.train_batch_size

    # print("drop_rate", args.drop_rate)
    # print("input_dim", args.input_dim)
    # print("size", args.bs)
    # print("output", args.output)
    print("train_batch_size", args.train_batch_size)
    print("epochs", args.epochs)


    train_X,train_y= los.generate_spell_data()
    # train_X = train_X.reshape((len(train_X),38, 150))

    start_time = time.time()
    with mlflow.start_run():
        mse,std = run_LR(train_X, train_y, epochs, batch_size)
        # mlflow.log_param("drop_rate", args.drop_rate)
        # mlflow.log_param("input_dim", args.input_dim)
        # mlflow.log_param("size", args.bs)
        # mlflow.log_param("output", args.output)
        mlflow.log_param("train_batch_size", args.train_batch_size)
        mlflow.log_param("epochs", args.epochs)
        # mlflow.log_param("acc", acc_score)
        # mlflow.log_param("f1", f1_score)
        # mlflow.log_param("roc_auc", roc_auc_score)
        mlflow.log_param("mse", mse)
        mlflow.log_param("std", std)


    timed = time.time() - start_time

    print("This model took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)





    


   






    

