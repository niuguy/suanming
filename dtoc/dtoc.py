"""
Preparing the training dataset of dtoc and provide the access API 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _pickle as pickle
import sys
import pandas as pd
import pyodbc
import datetime
import pandas.io.sql as psql
import time
import numpy as np
from gensim.models import Word2Vec
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#Generate the traing dataset from database of sql server
def dump_dtoc_datasets():
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=RBHDBSRED008;DATABASE=AI_DTOC;')
    # Query episodes data in batches
    t_start = time.time()
    chunk_size = 10000
    offset = 0
    dfs = []
    while True:
        query4episodes = '''SELECT [LOCAL_PATIENT_IDENTIFIER] as id,[CE_HOSPITAL_PROVIDER_SPELL_NUMBER] as sp_no,[CE_EPISODE_NUMBER] as ep_no,[CE_EPISODE_START_DATE_TIME] as start_date,[CEDS_DIAGNOSIS_01] as diag1
        ,[CEDS_DIAGNOSIS_02] as diag2,[CEDS_DIAGNOSIS_03] as diag3,[CEDS_DIAGNOSIS_04] as diag4, [CEDS_DIAGNOSIS_05] as diag5,
        [CEDS_DIAGNOSIS_06] as diag6,[CEDS_DIAGNOSIS_07] as diag7,[CEDS_DIAGNOSIS_08] as diag8,[CEDS_DIAGNOSIS_09] as diag9,
        [CEDS_DIAGNOSIS_10] as diag10,[CEDS_DIAGNOSIS_11] as diag11,[CEDS_DIAGNOSIS_12] as diag12, [HPS_START_DATE_TIME_HOSPITAL_PROVIDER_SPELL] as spell_time, 
        [HPS_DISCHARGE_DESTINATION_CODE_HOSPITAL_PROVIDER_SPELL] as dest_code, [HPS_AGE_AT_ADMISSION_DATE] as age 
        FROM [AI_DTOC].[dbo].[AI_DTOC_POPULATION_IP] ORDER BY spell_time ASC OFFSET %d ROWS FETCH NEXT %d ROWS ONLY '''% (offset, chunk_size)
        dfs.append(psql.read_sql(query4episodes, cnxn))
        offset += chunk_size   
    #     print 'the offset is', offset
        if len(dfs[-1]) < chunk_size:
            break
    df_ep = pd.concat(dfs)
    t_end = time.time()
    print('time eclapsed= ', t_end-t_start)

    # Read all dtoc data, inpatient_pathway_delays is a view
    query4dtocs = '''SELECT i.CE_HOSPITAL_PROVIDER_SPELL_NUMBER as sp_no,i.CE_EPISODE_NUMBER as ep_no
    FROM [AI_DTOC].[dbo].[AI_DTOC_POPULATION_IP] i
    INNEr JOIN [dbo].[PatientPathway] p ON i.CE_HOSPITAL_PROVIDER_SPELL_NUMBER = p.[Inpatient_EPR_ID_STR]
    AND i.[CE_EPISODE_NUMBER] = p.[EpisodeNumber]
    INNER JOIN [dbo].[tbDToC] d ON p.[Patient_Pathway_ID] = d.[PatientPathwayID]'''
    df_dtoc = psql.read_sql(query4dtocs, cnxn)

    # Label the dtoc dataset
    df_dtoc['is_dtoc'] = 1

    # Mark the dtoc in the episodes
    df_ep_m = pd.merge(df_ep, df_dtoc, how = 'left', on = ['sp_no', 'ep_no'], suffixes = ['','_r'])

    # If not dtoc, fill the label with 0
    df_ep_m['is_dtoc'].fillna(0, inplace = True)

    pickle.dump(df_ep_m, open('dataset/dtoc.pkl', 'wb'), -1)

    return df_ep_m

def get_diags_sequence(df):
    # all the spells , list all primary diagnoses in each spell
    df_seq = pd.DataFrame(columns=['id', 'diags'])

    # drop records that have no diagnosis
    df.dropna(subset = ['diag1'], inplace = True)
    grouped = df[['id','sp_no','ep_no','diag1']].sort_values(['sp_no', 'ep_no'], ascending = True).groupby(['id'])
    p_num = len(grouped)
    i = 0
    for name, group in grouped:
        diags = get_prim_spell_diags(group)
        df_seq.loc[i] = [name, diags]
        i += 1
    print('df_seq shape', df_seq.shape)
    pickle.dump(df_seq, open('dataset/diag_seq.pkl', 'wb'), -1)
    return df_seq    

def get_prim_diag_history(df,df_sample):
    # id->the patient id
    # filter
    # date -> the current visiting date
    his_diags = pd.DataFrame(None,columns=['diag_hist1','diag_hist2','diag_hist3','diag_hist4','diag_hist5'])
    for index,row in df_sample.iterrows():
        df_his = df[df['id']==row['id']]
        df_his = df_his[df_his['spell_time']<row['spell_time']]
        df_his = df_his.sort_values(['sp_no', 'ep_no'], ascending = True)
        diag_values =  df_his[df_his['diag1'].notna()].diag1.unique()
        if len(diag_values)>0:
            his_diags.loc[index, 'diag_hist5'] = diag_values[-1:]
        if len(diag_values)>1:
            his_diags.loc[index, 'diag_hist4'] = diag_values[-2:-1]
        if len(diag_values)>2:
            his_diags.loc[index, 'diag_hist3'] = diag_values[-3:-2]
        if len(diag_values)>3:
            his_diags.loc[index, 'diag_hist2'] = diag_values[-4:-3]
        if len(diag_values)>4:
            his_diags.loc[index, 'diag_hist1'] = diag_values[-5:-4]
        
    return his_diags
           


def get_prim_spell_diags(group):
    ret = set()
    for index, row in group.iterrows():
        ret.add(row['diag1'])
    return ','.join(ret)


def aggregate_diags(group):
    ret = set()
    for index, row in group.iterrows():
        for col in ['diag1','diag2','diag3','diag4','diag5','diag6', 
        'diag7', 'diag8', 'diag9','diag10', 'diag11', 'diag12']:
            if row[col]:
                ret.add(row[col])
    if len(ret)==0 :
        return None
    return ','.join(ret)



def spell_sample(n=40000):
    #the sample contains all dtoc data
    df = load_data(filename='dataset/dtoc_proc.pkl')
    #remove not-coded data
    df.dropna(subset=['diag1'], inplace= True)
    #set aside all the dtoc spells
    dtoc_spell_nos = df[df['is_dtoc']==1]['sp_no'].unique()
    dtoc_spell_size = len(dtoc_spell_nos)
    print('dtoc spells count:', dtoc_spell_size)

    n = dtoc_spell_size if int(n) < dtoc_spell_size else n #check the boundary of n
    ndtoc_spell_nos= df[~df['sp_no'].isin(dtoc_spell_nos)]['sp_no'].sample(n-dtoc_spell_size).unique()
    print('ndtoc spells count:', len(ndtoc_spell_nos))
    sample_spell_nos= np.concatenate((dtoc_spell_nos, ndtoc_spell_nos))


    spell_data = pd.DataFrame(columns = ['sp_no', 'diags', 'age', 'dest_code', 'los','is_dtoc'])
    sp_index = 0
    spell_grouped = df[df.sp_no.isin(sample_spell_nos)].groupby('sp_no')
    for name, group in spell_grouped:
        spell_data.loc[sp_index]= [name,aggregate_diags(group), group['age'].max(), group['dest_code'].values[0],group['los'].max(),group['is_dtoc'].max() ]
        sp_index += 1       
    spell_data = shuffle(spell_data)
    pickle.dump(spell_data, open('dataset/dtoc_los_spells.pkl', 'wb'), -1)
    return spell_data
        


def sample(n=40000, start='2010-01-01', end='2020-01-01', need_store = False):
    ## episode level samples
    df = load_data(filename='dataset/dtoc_proc.pkl') 
    df.dropna(subset=['diag1'], inplace= True)
    ## transfer admin_month_data 
    df['adm_month'] = pd.DatetimeIndex(df['start_date']).month
    #mast time period
    df['start_date'] = pd.to_datetime(df['start_date'])  
    period_mask = (df['start_date'] > start) & (df['start_date']<end)
    df = df.loc[period_mask]
    #Make sure n <= len(df)
    n = len(df) if int(n) > len(df) else int(n)
    df_dtoc = df[df['is_dtoc'] == 1]
    dtoc_size = len(df_dtoc)
    print('dtoc_size=', dtoc_size)
    df_ndtoc_sample = df[~df['id'].isin(df_dtoc.id)].sample(n-dtoc_size)
    #Concatenate and shuffle
    df_sample = pd.concat([df_ndtoc_sample, df_dtoc] )
    #Get history diagnoses
    df_hist_diags = get_prim_diag_history(df, df_sample)
    df_sample = pd.concat([df_sample, df_hist_diags], axis=1)
    df_sample = shuffle(df_sample)
    #store sample
    if need_store:
        pickle.dump(df_sample, open('dataset/dtoc_sample_40000_his.pkl', 'wb'), -1)
    
    return df_sample

def padding_single(value, dim, do_norm=True):
    rst = np.empty(150)
    if do_norm:
        rst.fill(int(value)/100)
    else:
        rst.fill(int(value))
    return rst



def vectorization_for_spell(df, wv_model, emb_dim):
      
    empty_vec = np.zeros(emb_dim)
    embeds = []
    for index, row in df.iterrows():
        ## add age
        row_embeds = np.empty(emb_dim)
        row_embeds.fill(row['age']/100)               
        for dig in row['diags'].split(','):
            if dig in wv_model.wv:
                row_embeds = np.concatenate((row_embeds, wv_model.wv[dig]), axis = None)      
        embeds.append(row_embeds)
    #save embeddings
    embeds_array = np.array(embeds)
    return embeds_array

def calculte_maxlen(train_arr, val_arr):
    max_len1 = np.max([len(a) for a in train_arr])
    max_len2 = np.max([len(a) for a in val_arr])
    return max([max_len1, max_len2])

def padding(arr, max_len):
    #pad to the max length
    rst = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in arr])
    return rst


def generate_spell_data():
    df = pickle.load(open('dataset/dtoc_los_spells.pkl', 'rb'))
    df.drop([1358], inplace= True) ## 1358 has too many diags
    train_df, val_df = train_test_split(df, test_size=0.08, random_state=2018)
    train_X = train_df[['diags','age']]
    train_y = train_df['los'].values  

    val_X = val_df[['diags','age']]
    val_y = val_df['los'].values

    EMBEDDING_MODEL_FILE = 'diag2vec.model'    
    wv_model = Word2Vec.load(EMBEDDING_MODEL_FILE)

    emb_dim = 150  
    train_X_embeds = vectorization_for_spell(train_X, wv_model,emb_dim)
  
    # pickle.dump(train_X_embeds, open('dataset/spell_train_x_emb.pkl', 'wb'), -1)
    # pickle.dump(train_y, open('dataset/spell_train_y.pkl', 'wb'), -1)
    val_X_embeds = vectorization_for_spell(val_X, wv_model, emb_dim)
    
    max_len = calculte_maxlen(train_X_embeds, val_X_embeds)
    print('max_len=', max_len)
    train_X_embeds = padding(train_X_embeds, max_len)
    val_X_embeds = padding(val_X_embeds, max_len)
    # pickle.dump(val_X_embeds, open('dataset/spell_val_x_emb.pkl', 'wb'), -1)
    # pickle.dump(val_y, open('dataset/spell_val_y.pkl', 'wb'), -1)
    return train_X_embeds, train_y, val_X_embeds, val_y




def load_data(filename='dataset/dtoc.pkl'):

    pd_dtoc = pickle.load(open(filename, 'rb'))

    
    return pd_dtoc


if __name__ == "__main__":
    # generate_episode_embeds(emb_model_file='diag2vec_150.model')
    #get_diags_sequence(load_data())
    sample(need_store=True)
    # df = load_data(filename='dataset/dtoc_proc.pkl')
    # df_sample = load_data("dataset/dtoc_sample_40000.pkl")[:10]
    # print(df_sample[['diag_hist1','diag_hist2','diag1','diag2']])
    # diag_hists = get_prim_diag_history(df, df_sample)
    # print(diag_hists)
    # print(len(diag_hists))
    # print(df_sample.columns)
