library(tidyverse)
library(class)
library(ggplot2)
library(data.table)

## Read spam data set and conver to X matrix and y vector we need for
## gradient descent.
KfoldCV<-function(X_mat,y_vec,ComputePredictions,fold_vec,K){
  data0 <- data.frame(X_mat,y_vec,fold_vec)
  error_vec <- rep(0,K)
  for (i in 1:K){
    test <- filter(data0,fold_vec==i)
    train <- filter(data0,fold_vec!=i)
    n0=ncol(data0)
    x_new <- test[,1:(n0-2)]
    y_new <- test[,n0-1]
    x_train <- train[,1:(n0-2)]
    y_train <- train[,n0-1]
    pred_new <- ComputePredictions(train=x_train,test=x_new,cl=y_train)
    for(j in 1:length(pred_new)){
      idx <- 0
      if(pred_new[j]!=y_new[j]){
        idx<-idx+1
      }
    }
    error_vec[i] <- idx/length(pred_new)
  }
  return(error_vec)
}

NNCV<-function(X_mat,y_vec,X_new,num_folds=5,max_neighbors=20){
  validation_fold_vec <- sample(rep(1:num_folds, l=nrow(X_mat)))
  error_mat <- matrix(nr=num_folds,nc=max_neighbors)
  for(i in 1:max_neighbors){
    knn0<-function(train, test, cl){
      y=knn(train, test, cl,i)
    }
    error_mat[,i]<-KfoldCV(X_mat,y_vec,knn0,validation_fold_vec,K=num_folds)
  }
  mean_error_vec<-apply(error_mat,2,mean)
  best_neighbors =which.min(mean_error_vec)
  pred<-knn(X_mat,X_new,y_vec[,1],k=best_neighbors)
  out<-list(one=pred,two=mean_error_vec,three=best_neighbors)
  return(out)
}

baseline<-function(train, test, cl){
  return(0)
}

KNNCV<-function(train, test, cl){
  y=knn(train, test, cl,out$three)
}

## download spam data set to local directory, if it is not present.
data<-read.table(url("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data"))

data<scale(data)
X_new<-data[1,1:57]#随便选的
X_mat<-data[,1:57]
y_vec<-data[58]

out<-NNCV(X_mat,y_vec,X_new,num_folds=10,max_neighbors=20)

ggplot(data.frame(matrix(1:20),out$two),aes(x=matrix.1.20., y=out.two, ))+geom_point()

test_fold_vec<-rep(1,4)
for(i in 2:4){
  out<-NNCV(X_mat,y_vec,X_new,num_folds=i,max_neighbors=20)
  test_fold_vec[i]<-out$one
}

x=matrix(1:4)
table(x,test_fold_vec)

summary(y_vec)
