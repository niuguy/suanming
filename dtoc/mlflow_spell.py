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
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def build_model(model_type):
    if model_type == "LSTM":
        return model_LSTM()
    elif model_type == "LSTM_ATTENTION":
        return model_LSTM_ATTENTION()
    elif model_type == "CNN":
        return model_CNN()
    elif model_type == "LR":
        return model_LR()
    else:
        raise NameError('Model name not defined')

def model_LSTM_ATTENTION():
    maxlen = 150
    inp = Input(shape=(38,maxlen))
    x = LSTM(128, return_sequences=True)(inp)
    x = LSTM(64, return_sequences=True)(x)
    
    x = Attention(maxlen)(x)
    x = Dense(64, activation="relu")(x)
    
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model

def model_LSTM():
  
    maxlen = 150
    inp = Input(shape=(38,maxlen))
    x = LSTM(128, return_sequences=True)(inp)
    x = LSTM(64, return_sequences=True)(x)
     
    # x = Attention(maxlen)(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model

def model_CNN():
    model = Sequential()
    model.add(Conv1D(128, 3,
                 activation='relu',
                 input_shape=(13, 150)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5)) ## To be discussed
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model

def model_LR():

    model = Sequential()
    model.add(Flatten())


    model.add(Dense(1,  # output dim is 2, one score per each class
                # kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                activation='sigmoid')) # input dimension = number of features your data h

                
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


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
    parser.add_argument("--model_type", help="Choose from LSTM,LSTM_ATTENTION, CNN, LR", action = 'store', default="LSTM", type = str)
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

    model = build_model(args.model_type)

    train_X,train_y,validate_X,validate_y= dtoc.generate_spell_data()
    train_X = train_X.reshape((len(train_X),38, 150))
    validate_X = validate_X.reshape((len(validate_y),38,150))

    start_time = time.time()
    with mlflow.start_run():
        acc_score, f1_score, roc_auc_score = run_model(model,train_X,train_y,validate_X,validate_y,batch_size,epochs)
        # mlflow.log_param("drop_rate", args.drop_rate)
        # mlflow.log_param("input_dim", args.input_dim)
        # mlflow.log_param("size", args.bs)
        # mlflow.log_param("output", args.output)
        mlflow.log_param("train_batch_size", args.train_batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("acc", acc_score)
        mlflow.log_param("f1", f1_score)
        mlflow.log_param("roc_auc", roc_auc_score)


    timed = time.time() - start_time

    print("This model took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)





    


   






    

