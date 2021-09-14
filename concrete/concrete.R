########################################################################################
########################################################################################
#18년 8월 자료분석 쿼리#


#DATA : Concrete data
#제출자 : 김음화

########################################################################################
########################################################################################
setwd("C://Users//lg//Desktop//concrete")

library(ggplot2)
library(h2o)
library(tree)
library(caret)

h2o.init()


dt <- read.csv("Concrete_Data.csv", header=T)
names(dt)


#---------------------------------------------------------------------------------------------------------
#연속형변수 분포 확인   
ggplot(dt, aes(Cement)) + 
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(Blast.Furnace.Slag)) + 
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20)) 

ggplot(dt, aes(Fly.Ash)) + 
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(Water)) + 
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(Superplasticizer )) + 
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(Coarse.Aggregate )) + 
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(Fine.Aggregate)) + 
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(Age)) + 
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20)) 

ggplot(dt, aes(Concrete.compressive.strength)) +  
  geom_histogram(position="identity", fill="skyblue", alpha=.5) +
  theme(axis.title=element_text(size=20))


#scatter plot
ggplot(dt, aes(x=Cement, y=Concrete.compressive.strength)) +
  geom_point(shape=10, size=2) +
  geom_smooth(method="loess") +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(x=Blast.Furnace.Slag, y=Concrete.compressive.strength)) +
  geom_point(shape=10, size=2) +
  geom_smooth(method="loess") +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(x=Fly.Ash, y=Concrete.compressive.strength)) +
  geom_point(shape=10, size=2) +
  geom_smooth(method="loess") +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(x=Water, y=Concrete.compressive.strength)) +
  geom_point(shape=10, size=2) +
  geom_smooth(method="loess") +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(x=Superplasticizer, y=Concrete.compressive.strength)) +
  geom_point(shape=10, size=2) +
  geom_smooth(method="loess") +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(x=Coarse.Aggregate, y=Concrete.compressive.strength)) +
  geom_point(shape=10, size=2) +
  geom_smooth(method="loess") +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(x=Fine.Aggregate, y=Concrete.compressive.strength)) +
  geom_point(shape=10, size=2) +
  geom_smooth(method="loess") +
  theme(axis.title=element_text(size=20))

ggplot(dt, aes(x=Age, y=Concrete.compressive.strength)) +
  geom_point(shape=10, size=2) +
  geom_smooth(method="loess") +
  theme(axis.title=element_text(size=20))




#---------------------------------------------------------------------------------------------------------
#data split

set.seed(1000)
idx <- sample(1:nrow(dt), nrow(dt)*0.7, replace=F)
tr <- dt[idx,]
te <- dt[-idx,]


#---------------------------------------------------------------------------------------------------------
#tree

my.con <- list(mincut=1, minsize=2, nmax=100, mindev=0.01)
tree <- tree(Concrete.compressive.strength~., data=tr, control=my.con)

summary(tree)
plot(tree);text(tree)

#pruning usnig cv(parameter tuning)
cv.trees <- cv.tree(tree, FUN=prune.tree, K=10)
plot(cv.trees)

min_idx <- which.min(cv.trees$dev)
best_pram <- cv.trees$size[min_idx]

prune.trees <- prune.tree(tree, best=best_pram)


plot(prune.trees);text(prune.trees, pretty=0) #age, cement, water interaction고려



#---------------------------------------------------------------------------------------------------------
#interaction 추가

form <- as.formula(Concrete.compressive.strength~.+Age*Cement)
tr_int <- model.matrix(form, data=tr)[,-1]
te_int <- model.matrix(form, data=te)[,-1]



#logistic
fit_glm <- glm(form, data=tr, family="gaussian")
summary(fit_glm)
p_glm <- predict(fit_glm, newdata = as.data.frame(te_int))
(rms_glm <- sqrt(mean((te$Concrete.compressive.strength - p_glm)^2)))

fit_glm_back <- glm(Concrete.compressive.strength~.+Age*Cement-Fine.Aggregate, data=tr, family="gaussian")
summary(fit_glm_back)

fit_glm_back <- glm(Concrete.compressive.strength~.+Age*Cement-Fine.Aggregate-Coarse.Aggregate, data=tr, family="gaussian")
summary(fit_glm_back)
p_glm2 <- predict(fit_glm_back, newdata = as.data.frame(te_int))
(rms_glm2 <- sqrt(mean((te$Concrete.compressive.strength - p_glm2)^2)))




#randomforest
control <- trainControl(method="cv", 
                        number=10, 
                        search="grid"
) 
fit <- train(x = tr[,-9], 
             y = tr[,9], 
             method="rf", 
             metric="RMSE", 
             ntree=300,
             tuneLength=10, 
             trControl=control
)

rf_fit <- fit$finalModel

p_rf <- predict(rf_fit, newdata = te[,-9])
(rms_rf <- sqrt(mean((te$Concrete.compressive.strength - p_rf)^2)))





fit2 <- train(x = tr_int, 
              y = tr[,9], 
              method="rf", 
              metric="RMSE", 
              ntree=300,
              tuneLength=10, 
              trControl=control,
              importance=T
)
plot(varImp(fit2, scale=F))

rf_fit2 <- fit2$finalModel
p_rf2 <- predict(rf_fit2, newdata = te_int)
(rms_rf2 <- sqrt(mean((te$Concrete.compressive.strength - p_rf2)^2)))



#DNN
tr_h2o <- data.frame(as.data.frame(tr_int), tr[,9])
colnames(tr_h2o) <- c(colnames(te_h2o), "Y")
tr_h2o <- as.h2o(tr_h2o)
te_h2o <- as.h2o(te_int)

dnn_fit <- h2o.deeplearning(x = 1:9,
                            y= 10,
                            training_frame = tr_h2o,
                            distribution = "AUTO",
                            epochs = 500,
                            activation = "RectifierWithDropout",
                            hidden_dropout_ratios = rep(0.2,3),
                            hidden = c(ncol(tr_h2o)*4, ncol(tr_h2o)*8, ncol(tr_h2o)*4),
                            max_w2 = 15,
                            l2 = 1e-3)
p_dnn <- h2o.predict(dnn_fit, newdata = te_h2o)
p_dnn <- as.data.frame(p_dnn$predict)
(rms_dnn <- sqrt(mean((te$Concrete.compressive.strength - p_dnn)^2)))

h2o.varimp_plot(dnn_fit)
