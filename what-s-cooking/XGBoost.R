#------------------------------------------------------------------------------#

# Title : XGBoost
# Author : EH

#------------------------------------------------------------------------------#
rm(list=ls())

#Packages
library(dplyr)
library(caret)
library(xgboost)

#Working directory
setwd("C://Users//PC2//Desktop//SKT//data")

load(file="Data_Handling.Rdata")
#------------------------------------------------------------------------------#

##data
#train_fin; test_fin
dim(train_fin); dim(test_fin)


#------------------------------------------------------------------------------
#convert data type for fitting xgboost
train_x <- train_fin[,-1]
train_y <- as.numeric(train_fin[,1])-1

test_x <- test_fin[,-1]
test_y <- as.numeric(test_fin[,1])-1


train_dt <- xgb.DMatrix(data =as.matrix(train_x) , label=train_y)
test_dt <- xgb.DMatrix(data=as.matrix(test_x), label=test_y)


xgb_tr <- model.matrix(cuisine~., data=train_fin)[,-1] #remove intercept term
xgb_te <- model.matrix(cuisine~., data=test_fin)[,-1]


#------------------------------------------------------------------------------
#fitting xgboost
params <- list(booster = "gbtree", 
               objective = "multi:softmax", 
               eta=0.1, 
               max_depth=25, 
               eval_metric = "merror",
               num_class=20
)

xgb.fit <- xgboost(params = params,
                   data = xgb_tr,
                   label = train_y,
                   nrounds = 300
)


#------------------------------------------------------------------------------
#results
p <- predict(xgb.fit, xgb_te, type='response')
tab <- table(test_y, p)
sum(diag(tab))/sum(tab)

#------------------------------------------------------------------------------
save(list=ls(), file="XGB_result.Rdata")
#------------------------------------------------------------------------------
