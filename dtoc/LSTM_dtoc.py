from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional, LSTM


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from dataset import dtoc
from gensim.models import Word2Vec
from keras import backend as K
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from sklearn import metrics

def vectorization(df, wv_model):
    empty_vec = np.zeros(150)
    embeds = []
    for index, row in df.iterrows():
        row_embeds = []
        for dig in df.columns:
            if row[dig]:
                row_embeds.append(wv_model.wv[row[dig]])
            else:
                row_embeds.append(empty_vec)
        embeds.append(row_embeds)
    return np.array(embeds)
    
def batch_gen(train_df, batch_size, wv_model):
    n_batches = math.ceil(len(train_df)/batch_size)
    while True:
        train_df = train_df.sample(frac=1.)
        for i in range(n_batches):
            train_df_batch = train_df.iloc[i*batch_size:(i+1)*batch_size, 4:16]
            icd_vectors = vectorization(train_df_batch, wv_model)
            yield icd_vectors, np.array(train_df['is_dtoc'][i*batch_size:(i+1)*batch_size])        




def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

                
def train(train_df, wv_model, validation_data):
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(12, 150)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[f1, 'acc'])

    batch_size = 128
    mg = batch_gen(train_df, batch_size, wv_model)
    model.fit_generator(mg, epochs=20,steps_per_epoch=1000,validation_data=validation_data,verbose=True)
    return model

def load_model():
    with open('lstm_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('lstm_weights.h5')
    return model


def digest_predict(sample_x):
    model = load_model()
    return model.predict(sample_x)

def evaluate(start_date, end_date, all_dtoc = False):
    df = dtoc.sample(40000,start_date,end_date)
    if all_dtoc:
        df = df[df['is_dtoc']==0]
    sample_x =vectorization(df.iloc[:, 4:16], Word2Vec.load("diag2vec.model"))
    model = load_model()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[f1, 'acc'])
    return model.evaluate(sample_x, df['is_dtoc'])

def predict_evaluate():
    model = load_model()
    df = dtoc.sample(40000)
    val_y = df['is_dtoc']
    sample_x =vectorization(df.iloc[:, 4:16], Word2Vec.load("diag2vec.model"))
    model = load_model()
    # model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[f1, 'acc'])
    pred_val_y = model.predict(sample_x)
    thresholds = []
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
        thresholds.append([thresh, res])
        print("F1 score at threshold {0} is {1}".format(thresh, res))

def main():
    df = dtoc.sample(40000,'2010-01-01','2018-01-01')
    train_df, val_df = train_test_split(df, test_size = 0.1)
    wv_model = Word2Vec.load("diag2vec.model")
    validation_data = (vectorization(val_df.iloc[:, 4:16], wv_model), np.array(val_df['is_dtoc']))
    model = train(train_df,wv_model, validation_data)
    
    ##save weights 
    model.save_weights('lstm_weights_before2018.h5')
    ##save structure
    with open('lstm_architecture_before2018.json', 'w') as f:
        f.write(model.to_json())

def model_visualization():
    model = load_model()
    print('Model summary:')
    print(model.summary())
    # plot_model(model, to_file='lstm_polt.png', show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    # main()
    model_visualization()
    # start_date = '2018-01-01'
    # for 
    # print(1,evaluate('2017-01-01','2017-02-01'))
    # print(2,evaluate('2017-02-01','2017-03-01'))
    # print(3,evaluate('2017-03-01','2017-04-01'))
    # print(4,evaluate('2017-04-01','2017-05-01'))
    # print(5,evaluate('2017-05-01','2017-06-01'))
    # print(6,evaluate('2017-06-01','2017-07-01'))
    # print(7,evaluate('2017-07-01','2017-08-01'))
    # print(8,evaluate('2017-08-01','2017-09-01'))
    # print(9,evaluate('2017-09-01','2017-10-01'))
    # print(10,evaluate('2017-10-01','2017-11-01'))
    # print(11,evaluate('2017-11-01','2017-12-01'))
    # print(12,evaluate('2017-12-01','2018-01-01'))
    # print(13,evaluate('2018-01-01','2018-02-01'))
    # print(14,evaluate('2018-02-01','2018-03-01'))
    # print(15,evaluate('2018-03-01','2018-04-01'))
    # print(16,evaluate('2018-04-01','2018-05-01'))
    # print(17,evaluate('2018-05-01','2018-06-01'))
    # print(18,evaluate('2018-06-01','2018-07-01'))
    # print(19,evaluate('2018-07-01','2018-08-01'))
    # print(20,evaluate('2018-08-01','2018-09-01'))
    # print(21,evaluate('2018-09-01','2018-10-01'))


    # print(9,evaluate('2018-01-01','2019-01-01', all_dtoc=False))