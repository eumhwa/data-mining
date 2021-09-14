#------------------------------------------------------------------------------#

# Title : Catboost
# Author : EH

#------------------------------------------------------------------------------#
rm(list=ls())

#Packages
library(dplyr)
library(caret)
library(catboost)

#Working directory
setwd("C://Users//PC2//Desktop//SKT//data")

load(file="Data_Handling.Rdata")
#------------------------------------------------------------------------------#

##data
#train_fin; test_fin
dim(train_fin); dim(test_fin)


#------------------------------------------------------------------------------
#createing validation data
tr <- train_fin[1:round(nrow(train_fin)*0.8),]
val <- train_fin[(round(nrow(train_fin)*0.8)+1):nrow(train_fin),]
te <- test_fin

tr$cuisine <- as.numeric(tr$cuisine)
val$cuisine <- as.numeric(val$cuisine)
te$cuisine <- as.numeric(te$cuisine)

#transforming catboost data type
cat_tr <- catboost.load_pool(tr[,-1], label=tr[,1])
cat_val <- catboost.load_pool(val[,-1], label=val[,1])
cat_te <- catboost.load_pool(te[,-1], label=te[,1])



#------------------------------------------------------------------------------
#fitting catboost
cat_params <- list(iterations = 500,
                   thread_count = 6,
                   loss_function = 'MultiClass',
                   learning_rate=0.1,
                   eval_metric ="Accuracy",
                   depth = 15,
                   rsm=1, #percentage of used features
                   boosting_type = "Plain", #for speeding up
                   bootstrap_type = "Bernoulli", #for speeding up
                   subsample=1)

model <- catboost.train(cat_tr, cat_val, cat_params)

#------------------------------------------------------------------------------
#result

p <- catboost.predict(model, cat_te, prediction_type ="Class")

tb_cat <- table(as.numeric(test_fin[,1]), p)
sum(diag(tb_cat))/sum(tb_cat)

#------------------------------------------------------------------------------
save(list=ls(), file="catboost_result.Rdata")
#------------------------------------------------------------------------------