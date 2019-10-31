library(h2o)
h2o.init()
library(tidyverse)


# import data
eds <- h2o.importFile('/home/jupyter/rich/dtoc_los.csv')

diags <- h2o.importFile('/home/jupyter/rich/diags.csv') 

diags$x <- as.character(diags$x)

# Tokenize 
diags.token <- h2o.tokenize(diags$x," ")

# Build word2vec model
diag2v.model <- h2o.word2vec(diags.token, sent_sample_rate = 0, epochs = 10)

# Sanity check
h2o.findSynonyms(diag2v.model, "Z349", count=5)

# Calculate the vector of each row 
diags.vecs <- h2o.transform(diag2v.model, diags$x, aggregate_method = "AVERAGE")


####
# preparing traing dataset

# 1. dtoc instances
eds_r <- read.csv('/home/jupyter/rich/dtoc_los.csv')

eds_r <- eds_r %>%  filter(!is.na(diag1) & diag1!="")

# The patient ids which have dtoc records
dtoc_ids <- eds_r %>% filter(is_dtoc==1) %>% .$id %>% unique()

non_dtoc_ids <- eds_r %>% filter(is_dtoc==0) %>% .$id %>% unique()


# The spell records that have dtoc
dtoc_sps <- 
  eds_r %>% 
  filter(is_dtoc==1) %>% 
  group_by(id) %>% 
  arrange(start_date) %>%
  summarise(is_dtoc=max(is_dtoc),sp_no=last(sp_no), age=last(age),start_date=last(start_date), dest_code=last(dest_code))

# The spell records that don't have dtoc
non_dtoc_sps<- 
  eds_r %>% 
  filter(is_dtoc==0) %>% 
  group_by(id) %>% 
  arrange(start_date) %>%
  summarise(is_dtoc=max(is_dtoc),sp_no=last(sp_no), age=last(age),start_date=last(start_date), dest_code=last(dest_code)) %>% 
  sample_n(26811)


gen_hist_diags <- function(eds, cur_sps, hist_num = 5) {
  # eds  the whole episodes dataset
  # cur_sps   the ongoing spells (1 spell = 1 admission) 
  # hist_num the number of history sps to be selected
  
  hist_diags <- eds_r %>% filter(id %in% cur_sps$id & !sp_no %in% cur_sps) %>% 
    #arrange(desc(ep_no)) %>%
    group_by(sp_no) %>% 
    arrange(desc(start_date)) %>%
    # distinct(id) %>%
    summarise(id = last(id), diag_last = last(diag1), start_date = last(start_date)) %>%
    group_by(id) %>%
    top_n(hist_num, wt=start_date) %>%
    summarise(diag_hist = paste(diag_last,collapse =',')) 
  
  hist_diags
}

dtoc_hist_diags <- gen_hist_diags(eds_r, dtoc_sps)

dtoc_sps_full <- left_join(dtoc_sps, dtoc_hist_diags, by='id')

non_dtoc_hist_diags <- gen_hist_diags(eds_r, non_dtoc_sps)

non_dtoc_sps_full <- left_join(non_dtoc_sps, non_dtoc_hist_diags, by='id')


sps_full <- rbind(dtoc_sps_full,non_dtoc_sps_full) %>% sample_n(nrow(sps_full)) 


eds_r %>% filter(id %in% non_dtoc_sps$id & !sp_no %in% non_dtoc_sps) %>% 
  #arrange(desc(ep_no)) %>%
  group_by(sp_no) %>% 
  arrange(desc(start_date)) %>%
 # distinct(id) %>%
  summarise(id = last(id), diag_last = last(diag1), start_date = last(start_date)) %>%
  group_by(id) %>%
  top_n(5, wt=start_date) %>%
  summarise(diag_hist = paste(diag_last,collapse =',')) 


sps_full_h2o <- as.h2o(sps_full)

sps_full_h2o$age <- sps_full_h2o$age /100

sps_full_h2o$dest_code <- sps_full_h2o$dest_code / 100

spell.hist.diags.token <- h2o.tokenize(sps_full_h2o$diag_hist,",")

spel.hist.diags.vecs <- h2o.transform(diag2v.model, spell.hist.diags.token, aggregate_method = "AVERAGE")

diags.data <- h2o.cbind(sps_full_h2o[c('is_dtoc','age','dest_code')], spel.hist.diags.vecs)

diags.data$is_dtoc <- as.factor(diags.data$is_dtoc)

diags.data.split <- h2o.splitFrame(diags.data, ratios = 0.8)

print("Build a basic GBM model")

h_train <- diags.data.split[[1]]

h_test <- diags.data.split[[2]]




### Build Other Baseline Models (DRF, GBM, DNN & XGB)
n_seed <- 12345

features <- c(names(spel.hist.diags.vecs), 'age', 'dest_code')
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


h2o.performance(model_gbm, newdata = h_test)

## Automl

aml <- h2o.automl(x = features, y = target,
                  training_frame = h_train,
                  max_models = 20,
                  seed = 1)

yhat_test <- h2o.predict(aml@leader, newdata = h_test)

