#------------------------------------------------------------------------------#

# Title : Data Handling
# Author : EH

#------------------------------------------------------------------------------#
rm(list=ls())

#Packages
library(jsonlite)
library(dplyr)
library(ggplot2)
library(lattice)
library(ranger)

#Working directory
setwd("C://Users//PC2//Desktop//SKT//data")

#------------------------------------------------------------------------------#
##1. Loading dataset
data <- fromJSON("train.json", flatten = TRUE)

#Checking the frequency of each class
plot_df <- as.data.frame(table(data$cuisine))
colnames(plot_df) <- c("cuisine", "frequency")

ggplot(data=plot_df, aes(x=cuisine, y=frequency)) +
  geom_bar(stat="identity") +
  coord_flip() +
  theme(text = element_text(size=20)) 


#------------------------------------------------------------------------------#
##2. Data preprocessing

#1) reshaping data : to long format
ingre_id <- rep(1:nrow(data), lengths(data$ingredients))
ingres <- unlist(data$ingredients)

dt <- data.frame(tmp_id=ingre_id, ingredients=ingres)

#text preprocessing
delete_ws <- function(x) gsub(" ", "", x, fixed = TRUE)

dt$ingredients <- delete_ws(dt$ingredients)
dt$ingredients <- tolower(dt$ingredients)
dt <- dt[!duplicated(dt),]

dt2 <- data.frame(id=data$id, cuisine=data$cuisine, tmp_id=1:nrow(data))
long_dt <- merge(dt, dt2, by.x="tmp_id") #long format data


#2) reshaping data : to wide format
long_dt <- long_dt[,c(3,2)]
long_dt <- data.frame(long_dt, cnt=1)

wide_dt <- reshape(long_dt, 
                   idvar = "id",  
                   timevar = "ingredients", 
                   direction = "wide")       #wide format data(TF matrix)

dat <- data.frame(cuisine=data$cuisine, wide_dt[,-1])
dat[is.na(dat)] <- 0


#------------------------------------------------------------------------------#
#3. Data split

set.seed(1000)
tr_idx <- sample(1:nrow(dat), round(nrow(dat)*0.8), replace=F)

train <- dat_fin[tr_idx,]
test <- dat_fin[-tr_idx,]

dim(train); dim(test)

#Removing sparse variable
col_sums <- apply(train[,-1], 2, sum)
summary(col_sums)

sparse_id <- which(col_sums <4) #threshold:1Q

train <- train[,-sparse_id]
test <- test[,-sparse_id]

dim(train); dim(test)


#------------------------------------------------------------------------------#
#4. Creating cluster variable

#calculating average point of class 
class_list <- as.character(unique(train$cuisine))
for(i in 1:length(class_list)){
  
  cl <- class_list[i]
  
  tmp_dt <- train[train$cuisine==cl,-1]
  tmp_colsum <- apply(tmp_dt, 2, mean)
  
  if(i==1) avg_cl <- tmp_colsum
  else avg_cl <- rbind(avg_cl, tmp_colsum)
}

#assigning the cluster_id
manh_dist <- function(x,y) sum(abs(x-y))
eucl_dist <- function(x,y) sum((x-y)^2)
create_cl_var <- function(data, avg_cluster){
  
  out_man <- c()
  out_euc <- c()
  for(i in 1:nrow(data)){
    v <- as.numeric(data[i,])
    
    dist_man <- as.numeric(apply(avg_cluster, 1, manh_dist, y=v))
    dist_euc <- as.numeric(apply(avg_cluster, 1, eucl_dist, y=v))
    
    id_man <- which.min(dist_man)
    id_euc <- which.min(dist_euc)
    
    out_man[i] <- id_man
    out_euc[i] <- id_euc
  }
  
  return(list(out_man=out_man,
              out_euc=out_euc)
  )
}

tr_cluster <- create_cl_var(train[,-1], avg_cl)
te_cluster <- create_cl_var(test[,-1], avg_cl)

#choice of measure [manhattan dist vs. euclidean dist]
label_chk <- class_list[tr_cluster$out_euc]
tb <- table(train[,1], label_chk)
sum(diag(tb))/sum(tb) #49.8%

heatmap(as.matrix(tb))
levelplot(t(as.matrix(tb)[c(nrow(as.matrix(tb)):1) , ]),  scales=list(x=list(rot=90)))

train$clust_id <- as.factor(tr_cluster$out_euc)
test$clust_id <- as.factor(te_cluster$out_euc)

dim(train); dim(test)


#------------------------------------------------------------------------------#
#5. Variable selection via ranger
g <- ranger(cuisine ~ .,
            data = train,
            num.trees = 1000,
            importance = "impurity_corrected",
            num.threads = 6  
)

imp <- importance_pvalues(g, method = "janitza")
sel <- rownames(imp[(imp[,2] <= 0.05),])
mufun <- as.formula(paste0("cuisine ~ ", paste0(sel, collapse = " + ")))


train_fin <- data.frame(train[,1], train[,sel])
test_fin <- data.frame(test[,1], test[,sel])

colnames(train_fin) <- c("cuisine", sel)
colnames(test_fin) <- c("cuisine", sel)

dim(train_fin); dim(test_fin)
#------------------------------------------------------------------------------#
save(list=ls(), file="Data_Handling.Rdata")
#------------------------------------------------------------------------------#