library(h2o)
h2o.init()
library(tidyverse)



# preparing traing dataset

# 1. dtoc instances
eds_r <- read.csv('/home/jupyter/rich/dtoc_los.csv')

eds_r <- eds_r %>%  filter(!is.na(diag1) & diag1!="")

# The patient ids which have dtoc records

NDTOC_SAMPLES <- 50000

# The spell records that have dtoc
dtoc_sps <- 
  eds_r %>% 
  filter(is_dtoc==1) %>% 
  group_by(id) %>% 
  arrange(start_date) %>%
  summarise(is_dtoc=max(is_dtoc),sp_no=last(sp_no), age=last(age),start_date=last(start_date), dest_code=last(dest_code))

# The spell records that don't have dtoc
ndtoc_sps<- 
  eds_r %>% 
  filter(is_dtoc==0) %>% 
  group_by(id) %>% 
  arrange(start_date) %>%
  summarise(is_dtoc=max(is_dtoc),sp_no=last(sp_no), age=last(age),start_date=last(start_date), dest_code=last(dest_code)) %>% 
  sample_n(NDTOC_SAMPLES)


add_hist_diags <- function(eds, cur_sps, hist_num = 5) {
  # eds  the whole episodes dataset
  # cur_sps   the ongoing spells (1 spell = 1 admission) 
  # hist_num the number of history sps to be selected
  
  # find history diags
  hist_diags <- eds_r %>% filter(id %in% cur_sps$id & !sp_no %in% cur_sps$sp_no) %>% 
    #arrange(desc(ep_no)) %>%
    group_by(sp_no) %>% 
    arrange(desc(start_date)) %>%
    # distinct(id) %>%
    summarise(id = last(id), diag_last = last(diag1), start_date = last(start_date)) %>%
    group_by(id) %>%
    top_n(hist_num, wt=start_date) %>%
    summarise(diag_hist = paste(diag_last,collapse =',')) 
  # Join current spells with history diags
  sps_full <- left_join(cur_sps, hist_diags, by='id')
  ## remove rows which have not history diags
  sps_full <- as.data.frame(sps_full) %>% filter(!is.na(diag_hist) & diag_hist!="") 
  sps_full
}

# Merge with original ones
dtoc_sps_full <- add_hist_diags(eds_r, dtoc_sps)

ndtoc_sps_full <- add_hist_diags(eds_r, ndtoc_sps)

## merge and shuffle
sps_full <- rbind(dtoc_sps_full,ndtoc_sps_full) 

sps_full <- sps_full %>% sample_n(nrow(sps_full))

sps_full <- as.h2o(sps_full)

#embedding and create train dataset

diag2v.model <- h2o.loadModel("/home/jupyter/work/suanming/Word2Vec_model_R_1572604381765_1")

do_encoding <- function(sps, diag2v.model){
  
  sps$age <- sps$age /100
  
  sps$dest_code <- sps$dest_code / 100
  
  diags.token <- h2o.tokenize(sps$diag_hist,",")
  
  diags.vecs <- h2o.transform(diag2v.model, diags.token, aggregate_method = "AVERAGE")
  
  data <- h2o.cbind(sps[c('is_dtoc','age','dest_code')], diags.vecs)
  
  data$is_dtoc <- as.factor(data$is_dtoc)
  data
}

sps_full <- do_encoding(sps_full, diag2v.model)

#### Training

# Train test split
sps.split <- h2o.splitFrame(sps_full, ratios = 0.8)

h_train <- sps.split[[1]]

h_test <- sps.split[[2]]

n_seed <- 12345
features <- c(names(sps_full[,2:103]))
target <- 'is_dtoc'

# Baseline Distributed Random Forest (DRF)
model_drf <- h2o.randomForest(x = features,
                              y = target,
                              training_frame = h_train,
                              model_id = "baseline_drf",
                              nfolds = 5,
                              seed = n_seed)

# Baseline Gradient Boosting Model (GBM)
model_gbm <- h2o.gbm(x = features,
                     y = target,
                     training_frame = h_train,
                     model_id = "baseline_gbm",
                     nfolds = 5,
                     seed = n_seed)

# Baseline Deep Nerual Network (DNN)
# By default, DNN is not reproducible with multi-core. You may get slightly different results here.
# You can enable the `reproducible` option but it will run on a single core (very slow).
model_dnn <- h2o.deeplearning(x = features, 
                              y = target, 
                              training_frame = h_train,
                              model_id = "baseline_dnn", 
                              nfolds = 5, 
                              seed = n_seed)

# predict on test
h2o.performance(model_gbm, newdata = h_test)

## Automl

aml <- h2o.automl(x = features, y = target,
                  training_frame = h_train,
                  max_models = 20,
                  seed = 1)

yhat_test <- h2o.predict(aml@leader, newdata = h_test)

aml@leader@model$cross_validation_metrics

