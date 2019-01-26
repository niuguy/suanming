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


def padding_single(value, dim, do_norm=True):
    rst = np.empty(dim)
    
    if do_norm:
        rst.fill(int(value)/100)
    else:
        rst.fill(int(value))
    return rst

def one_hot_transform(df, dim):
    enc = OneHotEncoder(handle_unknown='ignore')
    vectors =  enc.fit_transform(df).toarray()
    vectors = np.multiply(vectors, 2)
    rst = np.zeros((len(df),dim))
    rst[:,:vectors.shape[1]] = vectors
    rst = rst.reshape(len(df),1,dim)
    return rst

def one_hot_transform_single(df, dim=150):
    df = df.fillna('')
    df=df.values.reshape(-1,1)
    enc = OneHotEncoder(handle_unknown='ignore')
    vectors =  enc.fit_transform(df).toarray()

    # rst = np.zeros((len(df),dim))
    # rst[:,:vectors.shape[1]] = vectors
    # rst = rst.reshape(len(df),1,dim)
    return vectors

def one_hot_transform_trick(df, dim):
    df=df.reshape(-1,1)
    rst = np.zeros((len(df),dim))
    for i in range(len(df)):
        if df[i][0]=='':
            df[i][0]=0
        rst[i].fill(int(df[i][0])/100)
    
    rst = rst.reshape(len(df),1,dim)
    return rst


def vectorization(df, wv_model, emb_dim):
    
    empty_vec = np.zeros(emb_dim)
    embeds = []
    for index, row in df.iterrows():
        row_embeds = []
        for dig in df.columns:
            if row[dig]:
                row_embeds.append(wv_model.wv[row[dig]])
            else:
                row_embeds.append(empty_vec)        
        embeds.append(row_embeds)
    #save embeddings
    embeds_array = np.array(embeds)
    return embeds_array

def esemble_vectorization(df, wv_model_intra, wv_model_seq, emb_dim):
    
    empty_vec = np.zeros(emb_dim)
    embeds = []
    common_vector_count = 0
    for index, row in df.iterrows():
        row_embeds = []
        for dig in df.columns:
            code = row[dig]
            if isinstance(code, np.ndarray):
                code = code[0]
            if code in wv_model_intra.wv.vocab:
                if code in wv_model_seq.wv.vocab:
                    common_vector_count += 1
                    embed = np.mean([wv_model_intra.wv[code], wv_model_seq.wv[code]], axis=0)
                else:
                    embed = wv_model_intra.wv[code]
                row_embeds.append(embed)
            else:
                row_embeds.append(empty_vec)        
        embeds.append(row_embeds)
    #save embeddings
    embeds_array = np.array(embeds)
    print('the count of common vectors of the two datasets is:', common_vector_count)
    return embeds_array


def embed_X(df_in, wv_model_diags,wv_model_seqs , emb_type = 'm'):
    df_diags = df_in[['diag_hist1','diag_hist2','diag_hist3','diag_hist4','diag_hist5','diag2','diag3','diag4','diag5','diag6', 
    'diag7', 'diag8', 'diag9','diag10', 'diag11', 'diag12']]
    if emb_type=='m':
        df_diags_vec = esemble_vectorization(df_diags, wv_model_diags,wv_model_seqs, emb_dim)
    else:
        df_diags_vec = vectorization(df_diags, wv_model_diags, emb_dim)
    df_X = df_diags_vec
    for col in [ 'age', 'adm_code', 'gender','adm_month','is_oversea','dest_code']:
        df_other = df_in[col].fillna('')
        df_other_vec = one_hot_transform_trick(np.array(df_other), emb_dim)
        df_X = np.concatenate((df_X,df_other_vec), axis = 1)
    return df_X

def embed_for_predict():
    # This is a general example for prediction, the emb_model should not be loaded each time if making it as a server
    df_in = pd.DataFrame(columns=['diag_hist1','diag_hist2','diag_hist3','diag_hist4','diag_hist5','diag2','diag3','diag4','diag5','diag6', 
    'diag7', 'diag8', 'diag9','diag10', 'diag11', 'diag12', 'age', 'adm_code', 'gender','adm_month','is_oversea','dest_code'])
    emb_file_diags = "models/diag2vec_1_150.pkl"
    emb_file_seqs = "models/seq2vec_1_150.pkl"
    wv_model_seqs = Word2Vec.load(emb_file_seqs)
    wv_model_diags = Word2Vec.load(emb_file_diags)
    df_X = embed_X(df_in, wv_model_diags,wv_model_seqs)
    return df_X




def embed_for_train(data_file, emb_model, emb_dim, emb_type):
    ## data_file: the dataset to be embeded 
    df = dtoc.load_data(data_file)
    emb_file_diags = "models/diag2vec_"+str(emb_model)+"_"+str(emb_dim)+".pkl"
    emb_file_seqs = "models/seq2vec_"+str(emb_model)+"_"+str(emb_dim)+".pkl"
    wv_model_seqs = Word2Vec.load(emb_file_seqs)
    wv_model_diags = Word2Vec.load(emb_file_diags)
    df_X = embed_X(df, wv_model_diags, wv_model_seqs, emb_type)
    
    # procedure codes embedding
    # emb_file_procs = "models/proc2vec_"+str(emb_model)+"_"+str(emb_dim)+".pkl"
    # wv_model_procs = Word2Vec.load(emb_file_procs)
    # df_procs = df[['proc1','proc2','proc3','proc4','proc5','proc6', 
    # 'proc7', 'proc8', 'proc9','proc10', 'proc11', 'proc12']]
    # df_procs_vec = vectorization(df_procs,wv_model_procs, emb_dim)

    # df_diag_proc_vec = np.mean([df_diags_vec, df_procs_vec], axis=0)
    # df_X = np.concatenate((df_diags_vec,df_other_vec), axis = 1)
    # d_X = one_hot_dtoc(data_file)
    
    print('df_X shape:', df_X.shape)
    features_num = df_X.shape[1]
    
    df_y = df['is_dtoc'].values

    pickle.dump(df_X, open('dataset/embeds/dtoc_X_emb_' + str(features_num)+"_"+str(emb_model)+"_"+str(emb_dim)+"_"+str(emb_type)+".pkl", 'wb'), -1)
    pickle.dump(df_y, open('dataset/embeds/dtoc_y_emb_' + str(features_num)+"_"+str(emb_model)+"_"+str(emb_dim)+"_"+str(emb_type)+".pkl", 'wb'), -1)
    return  df_X, df_y

def one_hot(df, cols):
    df_new = pd.DataFrame()
    # print(df.columns)
    # for each in cols:
    #     dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
    #     df_new = pd.concat([df_new, dummies], axis=1)
    df_new = pd.get_dummies(df, columns=cols)    
    return df_new

def one_hot_dtoc(data_file):
     ## data_file: the dataset to be embeded   
    df = dtoc.load_data(data_file)
  
    df_X = one_hot(df, ['diag_hist1','diag_hist2','diag_hist3','diag_hist4','diag_hist5','diag2','diag3','diag4','diag5','diag6', 
     'diag7', 'diag8', 'diag9','diag10', 'diag11', 'diag12', 'age', 'adm_code', 'gender','adm_month','is_oversea','dest_code'])
    df_X = np.expand_dims(df_X, axis=2)
    print('df_X shape:', df_X.shape)
    features_num = df_X.shape[1]
    
    df_y = df['is_dtoc'].values

    pickle.dump(df_X, open('dataset/embeds/dtoc_X_emb_' + str(features_num)+"_1_1_m.pkl", 'wb'), -1)
    pickle.dump(df_y, open('dataset/embeds/dtoc_y_emb_' + str(features_num)+"_1_1_m.pkl", 'wb'), -1)
    return  df_X, df_y

        
def normal_hist_diags(data_file):
    df = dtoc.load_data(data_file)
    def get_first(x):
        if isinstance(x, np.ndarray):
            return x[0]
        return x
    df['diag_hist1'] = df['diag_hist1'].apply(lambda x:get_first(x))
    df['diag_hist2'] = df['diag_hist2'].apply(lambda x:get_first(x))
    df['diag_hist3'] = df['diag_hist3'].apply(lambda x:get_first(x))
    df['diag_hist4'] = df['diag_hist4'].apply(lambda x:get_first(x))
    df['diag_hist5'] = df['diag_hist5'].apply(lambda x:get_first(x))

    pickle.dump(df, open('dataset/dtoc_sample_40000_new.pkl', 'wb'), -1)

    print(df.head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", help="Raw data file", action = 'store', default="dataset/dtoc_sample_40000.pkl", type = str)
    parser.add_argument("--emb_model", help="Choose from CBOW(0),Skip-Gram(1)", action = 'store', default=1, type=str)
    parser.add_argument("--emb_dim", help="Embedding dimension", nargs='?', action='store', default=50, type=float)
    parser.add_argument("--save_emb", help="Save the embeds or not", nargs='?', action='store', default=True, type=bool)
    parser.add_argument("--emb_type", help="signle-embedding(s) or meta-embedding(m)", nargs='?', action='store', default='m', type=str)



    args = parser.parse_args()
    data_file = args.data_file
    emb_model = args.emb_model
    emb_dim = args.emb_dim
    emb_type = args.emb_type
 
    print('data_file:', data_file)
    print('emb_model:', emb_model)
    print('emb_dim:', emb_dim)
    print('emb_type:', emb_type)
    embed_for_train(data_file, emb_model, emb_dim, emb_type)
    # emb_file_seqs = "models/seq2vec_1_200.pkl"
    # wv_model_seqs = Word2Vec.load(emb_file_seqs)
    # print(wv_model_seqs.wv.vocab)
    # one_hot_dtoc(data_file)
    # normal_hist_diags(data_file)


