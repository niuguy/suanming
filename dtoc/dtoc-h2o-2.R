library(h2o)
h2o.init()
library(tidyverse)
library(ingredients)
library(DALEX)


# preparing traing dataset

# 1. dtoc instances
eds_r <- read.csv('/home/jupyter/rich/suanming/dtoc_proc.csv') %>% filter(!is.na(diag1) & diag1!="")


# The patient ids which have dtoc records

NDTOC_SAMPLES <- 10000

# The spell records that have dtoc
dtoc_sps <- 
  eds_r %>% 
  filter(is_dtoc==1) %>% 
  group_by(id) %>% 
  arrange(start_date) %>%
  summarise(is_dtoc=max(is_dtoc),sp_no=last(sp_no),
            age=last(age),start_date=last(start_date), 
            dest_code=last(dest_code), adm_code=last(adm_code),
            gender=last(gender),is_oversea=last(is_oversea))

# The spell records that don't have dtoc
ndtoc_sps<- 
  eds_r %>% 
  filter(is_dtoc==0) %>% 
  group_by(id) %>% 
  arrange(start_date) %>%
  summarise(is_dtoc=max(is_dtoc),sp_no=last(sp_no),
            age=last(age),start_date=last(start_date), 
            dest_code=last(dest_code), adm_code=last(adm_code),
            gender=last(gender),is_oversea=last(is_oversea)) 


add_hist_diags_procs <- function(eds, cur_sps, hist_num = 5) {
  # eds  the whole episodes dataset
  # cur_sps   the ongoing spells (1 spell = 1 admission) 
  # hist_num the number of history sps to be selected
  
  # find history diags
  hist_diags_procs <- eds_r %>% filter(id %in% cur_sps$id & !sp_no %in% cur_sps$sp_no) %>% 
    #arrange(desc(ep_no)) %>%
    group_by(sp_no) %>% 
    arrange(desc(start_date)) %>%
    # distinct(id) %>%
    summarise(id = last(id), diag_last_1 = last(diag1), diag_last_2 = last(diag2),
              diag_last_3 = last(diag3),diag_last_4 = last(diag4),diag_last_5 = last(diag5),
              diag_last_6 = last(diag6),diag_last_7 = last(diag7),diag_last_8 = last(diag8),
              diag_last_9 = last(diag9),diag_last_10 = last(diag10),diag_last_11 = last(diag11),
              diag_last_12 = last(diag12),proc_last_1 = last(proc1), proc_last_2 = last(proc2),
              proc_last_3 = last(proc3),proc_last_4 = last(proc4),proc_last_5 = last(proc5),
              proc_last_6 = last(proc6),proc_last_7 = last(proc7),proc_last_8 = last(proc8),
              proc_last_9 = last(proc9),proc_last_10 = last(proc10),proc_last_11 = last(proc11),
              proc_last_12 = last(proc12),start_date = last(start_date)) %>%
    group_by(id) %>%
    top_n(hist_num, wt=start_date) %>%
    summarise(diag_hist_1 = paste(diag_last_1,collapse =','),diag_hist_2 = paste(diag_last_2,collapse =','),
              diag_hist_3 = paste(diag_last_3,collapse =','),diag_hist_4 = paste(diag_last_4,collapse =','),
              diag_hist_5 = paste(diag_last_5,collapse =','),diag_hist_6 = paste(diag_last_6,collapse =','),
              diag_hist_7 = paste(diag_last_7,collapse =','),diag_hist_8 = paste(diag_last_8,collapse =','),
              diag_hist_9 = paste(diag_last_9,collapse =','),diag_hist_10 = paste(diag_last_10,collapse =','),
              diag_hist_11 = paste(diag_last_11,collapse =','),diag_hist_12 = paste(diag_last_12,collapse =','),
              proc_hist_1 = paste(proc_last_1,collapse =','),proc_hist_2 = paste(proc_last_2,collapse =','),
              proc_hist_3 = paste(proc_last_3,collapse =','),proc_hist_4 = paste(proc_last_4,collapse =','),
              proc_hist_5 = paste(proc_last_5,collapse =','),proc_hist_6 = paste(proc_last_6,collapse =','),
              proc_hist_7 = paste(proc_last_7,collapse =','),proc_hist_8 = paste(proc_last_8,collapse =','),
              proc_hist_9 = paste(proc_last_9,collapse =','),proc_hist_10 = paste(proc_last_10,collapse =','),
              proc_hist_11 = paste(proc_last_11,collapse =','),proc_hist_12 = paste(proc_last_12,collapse =',')) 
  # Join current spells with history diags
  sps_full <- left_join(cur_sps, hist_diags_procs, by='id')
  ## remove rows which have no history diags
  sps_full <- as.data.frame(sps_full) %>% filter(!is.na(diag_hist_1) & diag_hist_1!="") 
  sps_full
}


# Merge with original ones
dtoc_sps_full <- add_hist_diags_procs(eds_r, dtoc_sps)

ndtoc_sps_full <- add_hist_diags_procs(eds_r, ndtoc_sps)

## merge and shuffle
sps_full <- rbind(dtoc_sps_full,ndtoc_sps_full) 

# write.csv(sps_full, '/home/jupyter/rich/suanming/sps_full.csv')
# sps_full <- read.csv('/home/jupyter/rich/suanming/sps_full.csv')

sps_full <- sps_full %>% sample_n(nrow(sps_full))

#embedding and create train dataset

diag2v.model <- h2o.loadModel("/home/jupyter/rich/suanming/Word2Vec_model_R_1575475839112_1")

hist_diag2v.model <- h2o.loadModel("/home/jupyter/Word2Vec_model_R_1575923697613_1")

# proc2v.model <- h2o.loadModel("/home/jupyter/work/suanming/Word2Vec_model_R_1572678952167_391")


## embedding
do_embed_encoding <- function(sps, diag2v, hist_diag2v, proc2v){
  
  sps$age <- sps$age /100
  
  sps$dest_code <- sps$dest_code / 100
  
  sps <- sps %>% 
    unite(diag_hist_1, diag_hist_2, diag_hist_3,diag_hist_4,diag_hist_5,
                     diag_hist_6,diag_hist_7,diag_hist_8, diag_hist_9, diag_hist_10, 
                     diag_hist_11, diag_hist_12, col = 'diag_hists', sep=',') %>%
    unite(proc_hist_1, proc_hist_2, proc_hist_3,proc_hist_4,proc_hist_5,
          proc_hist_6,proc_hist_7,proc_hist_8, proc_hist_9, proc_hist_10, 
          proc_hist_11, proc_hist_12, col = 'proc_hists', sep=',')
    
  
  sps <- as.h2o(sps)
  diags.token <- h2o.tokenize(sps$diag_hists,",")

  diags.vecs <- h2o.transform(diag2v, diags.token, aggregate_method = "AVERAGE")
  
  if(!is.null(hist_diag2v)){
    hist_diags.vecs <- h2o.transform(hist_diag2v, diags.token, aggregate_method = "AVERAGE")
    diags.vecs <- (diags.vecs + hist_diags.vecs)/2
  }

#  proc.token <- h2o.tokenize(sps$proc_hists,",")
  
#  proc.vecs <- h2o.transform(proc2v, proc.token, aggregate_method = "AVERAGE")

  data <- h2o.cbind(sps[c('is_dtoc','age','dest_code')], diags.vecs)
 # data <- h2o.cbind(data, proc.vecs)

  data$is_dtoc <- as.factor(data$is_dtoc)
  data
}

## one_hot
do_onehot_encoding <- function(sps){
  
  sps$age <- sps$age /100
  
  sps$dest_code <- sps$dest_code / 100
  
  sps <- sps %>% 
    unite(diag_hist_1, diag_hist_2, diag_hist_3,diag_hist_4,diag_hist_5,
          diag_hist_6,diag_hist_7,diag_hist_8, diag_hist_9, diag_hist_10, 
          diag_hist_11, diag_hist_12, col = 'diag_hists', sep=',') %>% separate(diag_hists, sep=",", into =c("d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12", "d13","d14","d15","d16","d17","d18","d19","d20","d21","d22","d23","d24"))
  
  
  data <- as.h2o(sps)

  data$is_dtoc <- as.factor(data$is_dtoc)
  data
}

sps_full_one <- do_onehot_encoding(sps_full)


sps_full_e_h <- do_embed_encoding(sps_full, diag2v=diag2v.model, hist_diag2v = hist_diag2v.model)
sps_full_e <- do_embed_encoding(sps_full, diag2v=diag2v.model, hist_diag2v = NULL)



#Train test split(one-hot)
sps.split_one <- h2o.splitFrame(sps_full_one, ratios = 0.8)
# features_onehot <- c("age","dest_code","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12")

h_train <- sps.split_one[[1]]
h_test <- sps.split_one[[2]]
n_seed <- 12345
features <-  c("age","dest_code","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12")

# PCA preprocessing
pca_data <- sps_full_one[,c("age","dest_code","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12")]
diags.pca <- h2o.prcomp(training_frame = pca_data, transform = "STANDARDIZE",
                        k = 2, pca_method="Power", use_all_factor_levels=TRUE,
                        impute_missing=FALSE)
sps.split_one <- h2o.splitFrame(h2o.cbind(h2o.predict(diags.pca, pca_data), sps_full_one[,'is_dtoc']), ratios = 0.8)
h_train <- sps.split_one[[1]]
h_test <- sps.split_one[[2]]
features <- c('PC1','PC2')
target<-'is_dtoc'

model_gbm_one <- h2o.gbm(x = features,
                     y = "is_dtoc",
                     training_frame = h_train,
                     model_id = "baseline_gbm",
                     nfolds = 5,
                     # categorical_encoding = "OneHotExplicit",
                     seed = n_seed)


h2o.performance(model_gbm_one, newdata = h_test)

h2o.predict(model_gbm_one, newdata = h_test)


# Train test split(embedding)
sps.split <- h2o.splitFrame(sps_full_e, ratios = 0.8)
h_train <- sps.split[[1]]
h_test <- sps.split[[2]]
n_seed <- 12345
features <- names(sps_full_e)[2:103]
target <- 'is_dtoc'

# Baseline Deep Nerual Network (DNN)
# By default, DNN is not reproducible with multi-core. You may get slightly different results here.
# You can enable the `reproducible` option but it will run on a single core (very slow).
model_dnn <- h2o.deeplearning(x = features, 
                              y = target, 
                              training_frame = h_train,
                              model_id = "baseline_dnn", 
                              nfolds = 5, 
                              seed = n_seed)

h2o.performance(model_dnn, newdata = h_test)


# Baseline Gradient Boosting Model (GBM)
model_gbm_e <- h2o.gbm(x = features,
                     y = target,
                     training_frame = h_train,
                     model_id = "baseline_gbm",
                     nfolds = 5,
                     seed = n_seed)
h2o.performance(model_gbm_e, newdata = h_test)


# Custom Predict Function
custom_predict <- function(model, newdata) {
  newdata_h2o <- as.h2o(newdata)
  res <- as.data.frame(h2o.predict(model, newdata_h2o))
  return(as.numeric(res$predict))
}
explainer_gbm <- DALEX::explain(model = model_gbm
                                , data = as.data.frame(h_test)[, features],y = as.data.frame(h_test)[, target],predict_function = custom_predict,label = "H2O AutoML")
vi_gbm <- ingredients::feature_importance(explainer_gbm, type="difference")

# Baseline SVM

model_svm <- h2o.psvm(x = features, y = target, training_frame = h_train[1:1000,])


# Baseline Distributed Random Forest (DRF)
model_drf <- h2o.randomForest(x = features,
                              y = target,
                              training_frame = h_train,
                              model_id = "baseline_drf",
                              nfolds = 5,
                              seed = n_seed)
h2o.performance(model_drf, newdata = h_test)




# Baseline LR
model_lr <- h2o.glm(family= "binomial", 
        x= features,
        y=target, 
        training_frame=h_train, 
        lambda = 0, 
        compute_p_values = TRUE)

h2o.performance(model_lr, newdata = h_test)


## Automl

aml <- h2o.automl(x = features, y = target,
                  training_frame = h_train,
                  max_models = 20,
                  seed = 1)

yhat_test <- h2o.predict(aml@leader, newdata = h_test)

aml@leader@model$cross_validation_metrics


explainer_gbm <- DALEX::explain(model = model_gbm
                                   , data = as.data.frame(h_test)[, features],y = as.data.frame(h_test)[, target],predict_function = custom_predict,label = "H2O AutoML")
