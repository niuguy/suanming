library(h2o)
library(tidyverse)
h2o.init(max_mem_size = "32g")

diags <- h2o.importFile('/home/jupyter/rich/suanming/diags.csv') 

diags$x <- as.character(diags$x)

# Tokenize 
diags.token <- h2o.tokenize(diags$x," ")

# Build word2vec model
diag2v.model <- h2o.word2vec(diags.token, sent_sample_rate = 0, epochs = 10)

model_path <- h2o.saveModel(diag2v.model,path=getwd(), force=TRUE)

# History to vector
# Load all episodes
eds_r <- read.csv('/home/jupyter/rich/dtoc_proc.csv')

diags_hists <- 
  eds_r %>% 
  filter(!is.na(diag1) & diag1!="") %>%
  group_by(id) %>% 
  mutate(hist1s = paste0(diag1, collapse = ','), 
         hist2s= paste0(diag2, collapse = ',') ,
         hist3s= paste0(diag3, collapse = ','),
         hist4s= paste0(diag4, collapse = ','),
         hist5s= paste0(diag5, collapse = ','),
         hist6s= paste0(diag6, collapse = ','),
         hist7s= paste0(diag7, collapse = ','),
         hist8s= paste0(diag8, collapse = ','),
         hist9s= paste0(diag9, collapse = ','),
         hist10s= paste0(diag10, collapse = ','),
         hist11s= paste0(diag11, collapse = ','),
         hist12s= paste0(diag12, collapse = ',')) %>%
  select(id, hist1s, hist2s, hist3s, hist4s, hist5s, hist6s, hist7s, hist8s, hist9s, hist10s, hist11s, hist12s)


diags_hists <- h2o.importFile('/home/jupyter/rich/suanming/diags_hists_2.csv') 
diags_hists <- diags_hists[2:nrow(diags_hists),] 
diags_hists_token <- h2o.tokenize(as.character(diags_hists$C2)," ")
diag2v_hist_model <- h2o.word2vec(diags_hists_token, sent_sample_rate = 0, epochs = 10)

hist_model_path <- h2o.saveModel(diag2v_hist_model,path=getwd(), force=TRUE)

# One-hot encoding
eds_r <- read.csv('/home/jupyter/rich/suanming/dtoc_proc.csv')
diags.pca <- h2o.prcomp(training_frame =as.h2o(eds_r[c('diag1','diag2','diag3','diag4','diag5')][1:1000,]), transform = "STANDARDIZE",
                        k = 3, pca_method="Power", use_all_factor_levels=TRUE,
                        impute_missing=FALSE)

