library(h2o)
library(tidyverse)
h2o.init()

diags <- h2o.importFile('/home/jupyter/rich/diags.csv') 

diags$x <- as.character(diags$x)

# Tokenize 
diags.token <- h2o.tokenize(diags$x," ")

# Build word2vec model
diag2v.model <- h2o.word2vec(diags.token, sent_sample_rate = 0, epochs = 10)

model_path <- h2o.saveModel(diag2v.model,path=getwd(), force=TRUE)

