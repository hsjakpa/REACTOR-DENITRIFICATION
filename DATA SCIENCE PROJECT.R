library(MASS)
library(tree)
library(randomForest)
library(gbm)

###################################
#LOAD DATA 
#########################
NitDenit=read.csv(file="C:/Users/Horace/Documents/MSC ENV/SPRING 2020/DATA SCI/ND_data.csv")
head(NitDenit)
{
#################################################
#normalize data 
###############################################
normalize<-function(x) {return((x-min(x)) / (max(x)-min(x))) } 
ND_norm<- as.data.frame(lapply(NitDenit[,2:29], normalize))
head(ND_norm)

#============
#Organize
#============

names(ND_norm)
NitDenChar <-ND_norm[c("LogicND","DO..mg.L.","NO3...mg.L.","TIN..mg.L.")]
NitDenChar= na.omit(NitDenChar)
str(NitDenChar)
NitDenChar$LogicND[NitDenChar$LogicND == 0]<-'N'
NitDenChar$LogicND[NitDenChar$LogicND == 1]<-'D'
NitDenChar$LogicND <- as.factor(NitDenChar$LogicND)
colnames(NitDenChar) <- c("Y","DO","NO3","TIN")
str(NitDenChar)
head(NitDenChar)


#######50 iterations, 6 methods classification
set.seed(222)
#errmat = matrix(0,50,12)   #  50 iterations, 6 methods of classification
#for(i in 1:50)

#--------------------------------------------
  # Split data into Training and Test sample
  #--------------------------------------------
#############################

trainsampl<- sample(1:nrow(NitDenChar),0.7*nrow(NitDenChar))
traindata <- NitDenChar[trainsampl,] 
testdata<- NitDenChar[-trainsampl,]
Y_train<-NitDenChar$Y[trainsampl]
Y_test<-NitDenChar$Y[-trainsampl]

#-----------------------------------------------------------
# Knn
#-----------------------------------------------------------
ks = 1:50
library(class)
#testing error
KNNerrTest = 0
X_train = NitDenChar[trainsampl,-1]
X_test = NitDenChar[-trainsampl,-1]


for(j in 1:length(ks))
{
  predTest = knn(test=X_test,train=X_train,cl=Y_train,k=ks[j]) # knn function implements k-nearest neighbor
  KNNerrTest[j] = sum(Y_test!=predTest)/length(predTest)          # computes the test misclassification rate
}

k = which(KNNerrTest==min(KNNerrTest))
KNNerrTest = min(KNNerrTest)
predKnnTrain = knn(test=X_train,train=X_train,cl=Y_train,k=ks[j])
KNNerrTrain <- sum(Y_train!=predKnnTrain)/length(predKnnTrain)
KnnErr =c(KNNerrTrain, KNNerrTest)
KnnErr
summary(KnnErr)
k

########################################################
#-------------------------------------------------------------------------
# logistic regression
#-------------------------------------------------------------------------
lrmyData = glm(Y~.,data=NitDenChar,subset=Y_train,family=binomial)
trpred = (predict(lrmyData,type="response")) > 0.5
lrTrerr = mean((as.numeric(Y_train)-1)!=trpred)
tspred = (predict(lrmyData,newdata=testdata,type="response")) > 0.5
lrTserr = mean((as.numeric(Y_test)-1)!=tspred)
lrErr = c(lrTrerr, lrTserr)
lrErr
summary(lrmyData)

#--------------------------------------------------------------------
# add other methods QDA LDA Ridge Lasso	Elastic net to the loop here
#--------------------------------------------------------------------

#####################################
#-----LDA-----------
#---------------------############

library(MASS)
ldafit1<-lda(Y_train~.,data=X_train)
ldafit1
plot(ldafit1)
# Linear Discriminant Analysis
# LDA fit is available from the package MASS
library(MASS)  
lda.fit=lda(Y_train~., data=X_train[,2:3])
plot(lda.fit)
# lda.fit=lda( y ~ Xtrain[,1]+Xtrain[,2], CV=TRUE) # this fits LOOCV
lda.fit
plot(lda.fit) # plots the first two LDA directions
plot(lda.fit, dimen=1, type="both") # fit from lda only the first direction

lda.pred = predict (lda.fit)  # predicted values
names(lda.pred) # shows you the content of the object pred
lda.class = lda.pred$class # the predicted class levels

plot(X_train[,1],X_train[,3],col=as.numeric(Y_train),pch=20,xlab="DO",ylab="TIN",main = c("LDA")) # plot the results 
points(X_train[,1],X_train[,3],col=as.numeric(lda.class),pch=3) # overlay the predicted values over the actual 

table(lda.class,Y_train)  # the confusion matrix
mean(lda.class == Y_train) # correct classification rate
mean(lda.class != Y_train) # misclassification rate

#-------------------
#----QDA---------
# fit Quadratic Discriminatn Aanalysis (QDA)
qda.fit=qda( Y_train ~., data=X_train)
qda.fit
qda.pred=predict (qda.fit)  # predicted values
qda.class = qda.pred$class # the predicted class levels

plot(X_train[,1],X_train[,3],col=as.numeric(Y_train),pch=20,xlab="DO",ylab="TIN",main = c("QDA")) # plot the results 
points(X_train[,1],X_train[,3],col=as.numeric(qda.class),pch=3) # overlay the predicted values over the actual 

table(qda.class,Y_train)  # the confusion matrix
mean(qda.class == Y_train) # correct classification rate
mean(qda.class != Y_train) # misclassification rate

#-----------------------------------------------------------
# Decision Trees
#------------------------------------------------------------
library(tree)
set.seed(2)
treemyData=tree(Y~.,data=traindata)
cvmyData=cv.tree(treemyData,FUN=prune.misclass)
prunemyData=prune.misclass(treemyData,best=4)
treeTrpred=predict(prunemyData,type="class")
treeTspred=predict(prunemyData,X_test,type="class")
treeTrerr = mean(Y_train!=treeTrpred)
treeTserr = mean(Y_test!=treeTspred)
treeErr = c(treeTrerr, treeTserr)
treeErr

#-------------------------------------------------------------------------
# Random Forests
#-------------------------------------------------------------------------

set.seed(555)
train<- sample(1:nrow(NitDenChar),0.7*nrow(NitDenChar))
X_test=NitDenChar[-train,]
Y_test=NitDenChar$Y[-train]
Y_train = NitDenChar$Y[train]
library(randomForest)
colnames(NitDenChar)[1] <- "Y"

rf.ND = randomForest(Y~.,data=NitDenChar,subset=train,mtry=3)
rfmyData = randomForest(Y~.,data=NitDenChar,subset=train,mtry=2)
rfTrpred=predict(rfmyData,type="class")
rfTspred=predict(rfmyData,testdata,type="class")
rfTrerr = mean(Y_train!=rfTrpred)
rfTserr = mean(Y_test!=rfTspred)
rfErr = c(rfTrerr, rfTserr)
rfErr
importance(rfmyData)
varImpPlot(rfmyData)

#-------------------------------------------------------------------------
# Bagging
#-------------------------------------------------------------------------

set.seed(555)
train<- sample(1:nrow(NitDenChar),0.7*nrow(NitDenChar))
X_test=NitDenChar[-train,]
Y_test=NitDenChar$Y[-train]
Y_train = NitDenChar$Y[train]
#-------------------------------------------------------------------------
# Bagging class
#-------------------------------------------------------------------------
bag.ND = randomForest(Y~.,data=NitDenChar,subset=trainsampl,mtry=2)
bagTrpred=predict(bag.ND,type="class")
bagTspred=predict(bag.ND,X_test,type="class")
bagTrerr = mean(Y_train!=bagTrpred)
bagTserr = mean(Y_test!=bagTspred)
bagErr  = c(bagTrerr, bagTserr)
bagErr 
importance(bag.ND)

#-------------------------------------------------------------------------
# Boosting
#-------------------------------------------------------------------------
library(gbm)
y = as.numeric(NitDenChar$Y[Y_train])-1
myDataTr = data.frame(y,NitDenChar[Y_train,-1])
bsmyData=gbm(y~DO+TIN,data=myDataTr,distribution="bernoulli",n.trees=2300,interaction.depth=3)
bsTrpred=(predict(bsmyData,n.trees=2300,type="response")) > 0.4
bsTspred=(predict(bsmyData,n.trees=2300,testdata,type="response")) > 0.4
bsTrerr = mean((as.numeric(Y_train)-1)!=bsTrpred)
bsTserr = mean(as.numeric(Y_test)-1!=bsTspred)
bsErr = c(bsTrerr, bsTserr)
bsErr
summary(bsErr)

#-------------------------------------------------------------------------
# Summary of Results
#-------------------------------------------------------------------------
errmat = cbind(KnnErr, treeErr,bagErr,lrErr,rfErr,bsErr)
errmat = cbind(KnnErr, treeErr,bagErr,lrErr,rfErr,bsErr)
rownames(errmat) = c("Training","Test")
colnames(errmat) = c("KnnErr", "treeErr","bagErr","lrErr","rfErr","bsErr")
errmat
}
err = c(errmat)
# plots
labels = as.character(c( ("KnnTr"),("KnnTs"),
                         ("BagTr"),("BagTs"),
                         ("TreeTr"),("TreeTs"),
                         ("LRTr"),("LRTs"),
                         ("RFTr"),("RFTs"),
                         ("BSTr"),("BSTs")))

boxplot(err~labels,main="Classification Method Against Missclassifaction 
Error rate", ylab="Misclassification rate", xlab = "Method", ylim=c(-0.3,0.6))

