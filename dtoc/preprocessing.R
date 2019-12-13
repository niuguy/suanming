library(h2o)
library(tidyverse)
# Start H2O JVM
h2o.init(max_mem_size = "16g")

# Import all episodes
eds_r <- read.csv('/home/jupyter/rich/dtoc_proc.csv')

# Remove episodes that have no diagnose presented
eds_r <- eds_r %>%  filter(!is.na(diag1) & diag1!="")

