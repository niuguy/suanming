import os
import time
import argparse
import mlflow
import diag2vec
import _pickle as pickle
import numpy as np 
from numpy import zeros
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from gensim.models import Word2Vec
from sklearn import preprocessing
from dataset import dtoc
from mlflow import log_metric
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


def get_svm():
    clf = svm.SVC(gamma='auto')    
    return clf

def get_randomforest():
    clf = RandomForestRegressor(n_estimators=100, max_depth=2, random_state= 0)
    return clf

def get_LR():
    # clf =  LinearRegression()
    clf =  LogisticRegression()
    return clf 

def get_classifier(cl_type):
    if cl_type == 'SVM':
        return get_svm()
    elif cl_type == 'RF':
        return get_randomforest()
    elif cl_type == 'LR':
        return get_LR()
    else:
        raise NameError('Classifier not defined')


def run_classifier(clf, train_X, train_y, val_X, val_y, pred_threhold, clf_type, emb_dim):
    clf.fit(train_X, train_y)
    predict_val_origin_y = clf.predict(val_X)
    predict_val_y = (predict_val_origin_y>0.5).astype(int)
    acc = accuracy_score(val_y, predict_val_y)
    recall = recall_score(val_y, predict_val_y)
    f1 = metrics.f1_score(val_y, predict_val_y)
    roc_auc = metrics.roc_auc_score(val_y, predict_val_y)
    prec = precision_score(val_y, predict_val_y)

    fpr, tpr, thresholds = metrics.roc_curve(val_y, predict_val_origin_y, pos_label=1)
     ## save for later illustrating
    roc_result = pd.DataFrame(dict(fpr=fpr, tpr=tpr, thresholds = thresholds))
    pickle.dump(roc_result, open('dataset/roc_'+ str(clf_type)+"_"+str(embed_dim)+".pkl", 'wb'), -1)


    print("Classification report:\n%s\n"
      % (metrics.classification_report(val_y, predict_val_y))) 
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(val_y, predict_val_y))

    return prec, acc,recall,f1, roc_auc
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf_type", help="Choose from SVM,RF, LR", action = 'store', default="LR", type = str)
    parser.add_argument("--embed_model", help="the embedding model used, 0 for cbow, 1 for skip-gram", nargs='?', action='store', default=1, type=int)
    parser.add_argument("--emb_dim", help="Embedding dimension", nargs='?', action='store', default=200, type=int)
    parser.add_argument("--emb_type", help="signle-embedding(s) or meta-embedding(m)", nargs='?', action='store', default='m', type=str)
    parser.add_argument("--pred_threhold", help="Prediction threhold", nargs='?', action='store', default=0.5, type=float)

    args = parser.parse_args()

    embed_model = args.embed_model
    clf_type = args.clf_type
    embed_dim = args.emb_dim
    emb_type = args.emb_type
    pred_threhold = args.pred_threhold

    print("classifier_type", clf_type)
    print("pred_threthold", pred_threhold)
    print("embedding_model", embed_model)
    print("embedding_dimension", embed_dim)
    
    start_time = time.time()
    with mlflow.start_run():
        dtoc_X_emb = pickle.load(open("dataset/embeds/dtoc_X_emb_22_" + str(embed_model)+"_"+str(embed_dim)+"_"+str(emb_type)+".pkl", 'rb'))
        dtoc_y_emb = pickle.load(open('dataset/embeds/dtoc_y_emb_22_' + str(embed_model)+"_"+str(embed_dim)+"_"+str(emb_type)+".pkl", 'rb'))
        dtoc_X_emb = dtoc_X_emb.reshape((dtoc_X_emb.shape[0], dtoc_X_emb.shape[1]*dtoc_X_emb.shape[2]))
        train_X,validate_X,train_y, validate_y= train_test_split(dtoc_X_emb, dtoc_y_emb, test_size=0.08, random_state=2018)
 
        clf = get_classifier(clf_type)

        prec, acc,recall, f1, roc_auc = run_classifier(clf,train_X,train_y,validate_X,validate_y, pred_threhold,clf_type, embed_dim)

        mlflow.log_param("classifier_type", clf_type)
        mlflow.log_param("pred_threhold", pred_threhold)
        mlflow.log_param("embedding_model", embed_model)
        mlflow.log_param("embedding_dimension", embed_dim)
        mlflow.log_metric("prec", prec)
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

    timed = time.time() - start_time
    print("This classifier took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)





    


   






    

