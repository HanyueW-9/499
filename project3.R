library(tidyverse)

logloss<-function(x){
  y=log(1+exp(-x))
  return(y)
}

NNetOneSplit<-function(X.mat,y.vec,max.epochs,step.size,n.hidden.units,is.subtrain){
  dataX=data.frame(X.mat,is.subtrain)
  dataY=data.frame(y.vec,is.subtrain)
  X.subtrain=filter(dataX,is.subtrain==1)[,1:ncol(X.mat)]
  X.validation=filter(dataX,is.subtrain==0)[,1:ncol(X.mat)]
  Y.subtrain=filter(dataY,is.subtrain==1)[,1]
  Y.validation=filter(dataY,is.subtrain==0)[,1]
  V.mat<-matrix(rnorm(ncol(X.mat)*n.hidden.units,sd=0.1),ncol(X.mat),n.hidden.units)
  w.vec<-rnorm(n.hidden.units,sd=0.1)
  w.vect<-t(w.vec)
  w.vec<-t(w.vect)
  loss.values<-matrix( ,nrow = 2)
  for(k in 1:max.epochs){
    for(j in 1:nrow(X.subtrain)){
      pred<- as.matrix(X.subtrain[j,]) %*%V.mat %*%w.vec
      V.mat<-V.mat-as.numeric(step.size*(pred-Y.subtrain[j]))*(t(as.matrix(X.subtrain[j,]))%*%w.vect)
      w.vec<-w.vec-t(as.numeric(step.size*(pred-Y.subtrain[j]))*(as.matrix(X.subtrain[j,])%*%V.mat))
    }
    loss1<-as.vector(as.matrix(Y.subtrain)*as.matrix(X.subtrain)%*%V.mat %*%w.vec)
    loss1<-mean(as.vector(unlist(lapply(loss1,logloss))))
    loss2<-as.vector(as.matrix(Y.validation)*as.matrix(X.validation)%*%V.mat %*%w.vec)
    loss2<-mean(as.vector(unlist(lapply(loss2,logloss))))
    loss<-c(loss1,loss2)
    loss.values<-cbind(loss.values,loss)
  }
  loss.values<-loss.values[,2:(max.epochs+1)]
  out<-list(one=loss.values,two=V.mat,three=w.vec)
}

data <- read.table(url("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data"))
y <- data[,58]
y1<-(y-0.5)*2
data0<-data[,1:57]
data0<-scale(data0)
data<-data.frame(data0,y1)
is.train=c(rep(1,3680),rep(0,921))
is.train=sample(is.train,4601)
is.subtrain=c(rep(1,2208),rep(0,1472))
is.subtrain=sample(is.subtrain,3680)
max.epochs=10
data<-data.frame(data,is.train)
data.test<-filter(data,is.train==0)
data.train<-filter(data,is.train==1)
X.mat<-data.train[,1:57]
y.vec<-data.train[,58]
step.size=0.001
n.hidden.units=3
out<-NNetOneSplit(X.mat,y.vec,max.epochs,step.size,n.hidden.units,is.subtrain)

loss.values<-out$one
plot(c(1:10),loss.values[1,])
plot(c(1:10),loss.values[2,])

best_epochs<-9
is.subtrain<-c(rep(1,3680))
out<-NNetOneSplit(X.mat,y.vec,max.epochs,step.size,n.hidden.units,is.subtrain)
V.mat<-out$two
w.vec<-out$three
X_test<-data.test[,1:57]
y_test<-data.test[58]
pred<- as.matrix(X_test) %*%V.mat %*%w.vec
y0<-y_test-sign(pred)
table(y0)
828/921
