import os
import time
import argparse
import numpy as np 
from numpy import zeros
import pandas as pd 
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Flatten
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
from dataset import dtoc
from keras.callbacks import TensorBoard
import mlflow
from mlflow import log_metric
from sklearn.metrics import precision_score,accuracy_score, recall_score, confusion_matrix, roc_curve, auc
import diag2vec
import _pickle as pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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


def build_model(model_type, dim, emb_dim):
    if model_type == "LSTM":
        return model_LSTM(dim, emb_dim)
    elif model_type == "LSTM_ATTENTION":
        return model_LSTM_ATTENTION()
    elif model_type == "CNN":
        return model_CNN(dim, emb_dim)
    elif model_type == "LR":
        return model_LR()
    else:
        raise NameError('Model name not defined')

def model_LSTM_ATTENTION():
    maxlen = 13
    inp = Input(shape=(maxlen,150,))
    x = LSTM(128, return_sequences=True)(inp)
    x = LSTM(64, return_sequences=True)(x)
    
    x = Attention(maxlen)(x)
    x = Dense(64, activation="relu")(x)
    
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model

def model_LSTM(dim, emb_dim):
  

    inp = Input(shape=(dim,emb_dim,))
    x = LSTM(128, return_sequences=True)(inp)
    x = LSTM(64, return_sequences=True)(x)
    
    # x = Attention(maxlen)(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    return model

def model_CNN(dim, emb_dim):
    model = Sequential()
    model.add(Conv1D(128, 3,
                 activation='relu',
                 input_shape=(dim, emb_dim)))
    # model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPool1D())
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

def plot_roc(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            lw=1.0, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



def run_model(model, train_X, train_y, val_X, val_y, bsize, eps, pred_threhold, model_type, embed_dim):

    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(train_X, train_y, batch_size=bsize, epochs=eps, validation_data = (val_X, val_y), callbacks=[tbCallBack])

    # model_json = model.to_json()
    # with open("models/model.json", "w") as json_file: json_file.write(model_json)
    model.save("models/deep_pred_model.h5")
    # print("Saved model to disk")


    predict_val_origin_y = model.predict(val_X, batch_size=bsize, verbose=1)
    predict_val_y = (predict_val_origin_y>0.5).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(val_y, predict_val_origin_y, pos_label=1)


    ## save for later illustrating
    roc_result = pd.DataFrame(dict(fpr=fpr, tpr=tpr, thresholds = thresholds))    
    prec = precision_score(val_y, predict_val_y)
    acc = accuracy_score(val_y, predict_val_y)
    recall = recall_score(val_y, predict_val_y)
    f1 = metrics.f1_score(val_y, predict_val_y)
    roc_auc = metrics.roc_auc_score(val_y, predict_val_y)

    print("Classification report:\n%s\n"
      % (metrics.classification_report(val_y, predict_val_y)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(val_y, predict_val_y))

    return prec,acc,recall,f1, roc_auc, roc_result
    
def predict_one_record(model, input_X):
    return model.predict(input_X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="Choose from LSTM,LSTM_ATTENTION, CNN, LR", action = 'store', default="CNN", type = str)
    parser.add_argument("--pred_threhold", help="Prediction threhold", nargs='?', action='store', default=0.5, type=float)
    parser.add_argument("--input_dim", help="Input dimension for the network.", action='store', nargs='?', default=20, type=int)
    parser.add_argument("--bs", help="Number of rows or size of the tensor", action='store', nargs='?', default=1000, type=int)
    parser.add_argument("--output", help="Output from First & Hidden Layers", action='store',  nargs='?', default=64, type=int)
    parser.add_argument("--train_batch_size", help="Training Batch Size", nargs='?', action='store', default=1024, type=int)
    parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20, type=int)
    parser.add_argument("--embed_model", help="the embedding model used, 0 for cbow, 1 for skip-gram", nargs='?', action='store', default=1, type=int)
    parser.add_argument("--emb_dim", help="Embedding dimension", nargs='?', action='store', default=150, type=int)
    parser.add_argument("--emb_type", help="signle-embedding(s) or meta-embedding(m)", nargs='?', action='store', default='m', type=str)

    args = parser.parse_args()

    input_dim = args.input_dim
    bs = args.bs
    output = args.output
    epochs = args.epochs
    batch_size = args.train_batch_size
    embed_model = args.embed_model
    model_type = args.model_type
    embed_dim = args.emb_dim
    emb_type = args.emb_type
    pred_threhold = args.pred_threhold

    print("model_type", model_type)
    print("pred_threthold", pred_threhold)
    print("embedding_model", embed_model)
    print("embedding_dimension", embed_dim)
    print("train_batch_size", batch_size)
    print("epochs", epochs)
    print('emb_type:', emb_type)
    
    start_time = time.time()
    with mlflow.start_run():
        dtoc_X_emb = pickle.load(open("dataset/embeds/dtoc_X_emb_22_" + str(embed_model)+"_"+str(embed_dim)+"_"+str(emb_type)+".pkl", 'rb'))
        dtoc_y_emb = pickle.load(open('dataset/embeds/dtoc_y_emb_22_' + str(embed_model)+"_"+str(embed_dim)+"_"+str(emb_type)+".pkl", 'rb'))
        train_X,validate_X,train_y, validate_y= train_test_split(dtoc_X_emb, dtoc_y_emb, test_size=0.08, random_state=2018)
 
        input_dim = train_X.shape[1]
        input_emb_dim = embed_dim
        model = build_model(model_type, input_dim, input_emb_dim)

        prec,acc,recall,f1, roc_auc, roc_result= run_model(model,train_X,train_y,validate_X,validate_y,batch_size,epochs, pred_threhold,model_type, embed_dim)

        roc_file_name = 'dataset/roc_'+ str(model_type)+"_"+str(embed_dim)+".pkl"
        pickle.dump(roc_result, open(roc_file_name, 'wb'), -1)


        mlflow.log_param("model_type", model_type)
        mlflow.log_param("pred_threhold", pred_threhold)
        mlflow.log_param("embedding_model", embed_model)
        mlflow.log_param("embedding_type", emb_type)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("input_emb_dim", input_emb_dim)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("prec", prec)
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_artifact(roc_file_name)

    timed = time.time() - start_time
    print("This model took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)





    


   






    

