install.packages('neuralnet')
install.packages('boot')
install.packages('plyr')
install.packages('ROSE')
install.packages('party')
install.packages("EvaluationMeasures")
install.packages('randomForest')
install.packages('caret')
install.packages('e1071')
install.packages('LiblineaR')
install.packages('e1071')
install.packages('ROCR')

# Read Data after preprocessing
bank_data<-read.csv("cleaned.csv",head=TRUE)
attach(bank_data)
# Check is there any missing data
apply(bank_data,2,function(x) sum(is.na(x)))
set.seed(123)

# 5-folder cv
shuffle_data = bank_data[sample(nrow(bank_data)),]
folders = cut(seq(1,nrow(shuffle_data)),breaks=5, labels=FALSE)

# Build function to calculate MCC and perform regularization:
library("EvaluationMeasures")
MCC = function(real,predict)
{
  TPFN=table(real,predict)
  TP=as.numeric(TPFN[2,2])
  TN=as.numeric(TPFN[1,1])
  FP=as.numeric(TPFN[1,2])
  FN=as.numeric(TPFN[2,1])  
  return (round(((TP*TN)-(FP*FN))/(sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))),4))
}

#Normalization function
normalize = function(x) {
  return ((x-min(x))/(max(x-min(x))))
}

# Building F-measure function:
precision = function(real,predict)
{
  TPFN=table(real,predict)
  TP=TPFN[2,2]
  TN=TPFN[1,1]
  FP=TPFN[1,2]
  FN=TPFN[2,1] 
  pre = TP/(TP+FP)
  return (pre)
} 

recall = function(real,predict)
{
  TPFN=table(real,predict)
  TP=TPFN[2,2]
  TN=TPFN[1,1]
  FP=TPFN[1,2]
  FN=TPFN[2,1] 
  pre = TP/(TP+FN)
  return (pre)
}

F_measure = function(real,predict)
{
  f_measure = 2* precision(real,predict)*recall(real,predict)/(precision(real,predict)+recall(real,predict))
  
  return (f_measure)
}

# building accuracy function:
accuracy= function(real,predict)
{
  TPFN=table(real,predict)
  TP=as.numeric(TPFN[2,2])
  TN=as.numeric(TPFN[1,1])
  FP=as.numeric(TPFN[1,2])
  FN=as.numeric(TPFN[2,1])
  acc = (TP+TN)/(TP+FP+TN+FN)
  return (acc)
}

# import library for ROC curve plotting:
# Plot ROC curve 
library("ROCR")

# Model 1: Decision tree:
library('party')
mcc_dt_before=rep(0,5)
mcc_dt_after=rep(0,5)
F_dt_before=rep(0,5)
F_dt_after=rep(0,5)
a_dt_before=rep(0,5)
a_dt_after=rep(0,5)
for(i in 1:5)
{
  #
  testIndexes = which (folders==i)
  testData = shuffle_data[testIndexes,]
  trainData = shuffle_data[-testIndexes,]
  table(testData$y)
  
  # before processing the imbalance:
  decisionTreeModel_before <- ctree(y ~ ., data = trainData)
  pred_1 = predict(decisionTreeModel_before,testData[,-16])
  table(testData$y,pred_1)
  
  # Deal with the imbalance:
  library("ROSE")
  After_trainData = ovun.sample(y ~ ., data = trainData, method = "both", p=0.5,N=36200, seed = 123)$data
  
  # after processing the imbalance:
  decisionTreeModel_after <- ctree(y ~ ., data = After_trainData)
  pred_2 = predict(decisionTreeModel_after,testData[,-16])
  table(testData$y,pred_2)
  
  # calculate MCC and F-measure for both models:
  mcc_dt_before[i]=MCC(testData$y,pred_1)
  mcc_dt_after[i]=MCC(testData$y,pred_2)
  F_dt_before[i]=F_measure(testData$y,pred_1)
  F_dt_after[i]=F_measure(testData$y,pred_2)
  a_dt_before[i]=accuracy(testData$y,pred_1)
  a_dt_after[i]=accuracy(testData$y,pred_2)
  
}
sum(mcc_dt_after)/5
sum(mcc_dt_before)/5
sum(F_dt_before)/5
sum(F_dt_after)/5
sum(a_dt_before)/5
sum(a_dt_after)/5


# Model 2: Random Forest: 
library(randomForest)
mcc_rf_before=rep(0,5)
mcc_rf_after=rep(0,5)
F_rf_before=rep(0,5)
F_rf_after=rep(0,5)
a_rf_before=rep(0,5)
a_rf_after=rep(0,5)
for(i in 1:5)
{
  testIndexes = which (folders==i)
  testData = shuffle_data[testIndexes,]
  trainData = shuffle_data[-testIndexes,]
  table(testData$y)
  
  #before processing the imbalance:
  rf_before <- randomForest(y~., data=trainData)
  print(rf_before)
  pred_3=predict(rf_before,testData[,-16])
  table(testData$y,pred_3)
  
  #after processing the imbalance:
  After_trainData = ovun.sample(y ~ ., data = trainData, method = "both", p=0.5,N=36200, seed = 123)$data
  rf_after <- randomForest(y~., data=After_trainData)
  print(rf_after)
  pred_4=predict(rf_after,testData[,-16])
  table(testData$y,pred_4)
  mcc_rf_before[i]=MCC(testData$y,pred_3)
  mcc_rf_after[i]=MCC(testData$y,pred_4)
  F_rf_before[i]=F_measure(testData$y,pred_3)
  F_rf_after[i]=F_measure(testData$y,pred_4)
  a_rf_before[i]=accuracy(testData$y,pred_3)
  a_rf_after[i]=accuracy(testData$y,pred_4)
}
sum(mcc_rf_before)/5
sum(mcc_rf_after)/5
sum(F_rf_before)/5
sum(F_rf_after)/5
sum(a_rf_before)/5
sum(a_rf_after)/5


# Performing feature selection for random forest:  
library(caret)
library(e1071)
# Feature selection for random forest after processing the imbalance:
importance(rf_after)

mcc_rffs=rep(0,5)
F_rffs=rep(0,5)
a_rffs=rep(0,5)
for(i in 1:5)
{
  testIndexes = which (folders==i)
  testData = shuffle_data[testIndexes,]
  trainData = shuffle_data[-testIndexes,]
  table(testData$y)
  names(After_trainData)
  After_trainData = ovun.sample(y ~ ., data = trainData, method = "both", p=0.5,N=36200, seed = 123)$data
  
  rf_fs <- randomForest(y~duration+month+balance+age+day, data=After_trainData)
  summary(rf_fs)
  names(testData)
  predfs=predict(rf_fs,testData[,c(1,2,4,6,8,10,11,12,14)])
  summary(predfs)
  table(testData$y,predfs)
  length(testData$y)
  mcc_rffs[i]=MCC(testData$y,predfs)
  F_rffs[i]=F_measure(testData$y,predfs)
  a_rffs[i]=accuracy(testData$y,predfs)
  #     no    yes           no  yes  
  # no  7710  273     no  7190  793
  # yes  551  509     yes  242  818
}
sum(mcc_rffs)/5
sum(F_rffs)/5
sum(a_rffs)/5

#logistic regression before normalization
mcc_lr1_before=rep(0,5)
mcc_lr1_after=rep(0,5)
F_lr1_before=rep(0,5)
F_lr1_after=rep(0,5)
a_lr1_before=rep(0,5)
a_lr1_after=rep(0,5)
auc_lr1_before=rep(0,5)
auc_lr1_after=rep(0,5)
for(i in 1:5)
{
  testIndexes = which (folders==i)
  testData = shuffle_data[testIndexes,]
  trainData = shuffle_data[-testIndexes,]
  table(testData$y)
  
  ## convert factor variable into numerical type  
  numericTrainData= data.frame(sapply(trainData[,-16],as.numeric))
  numericTestData= data.frame(sapply(testData[,-16],as.numeric))
  lrnumericTrainData=cbind(numericTrainData,trainData[,'y'])
  lrnumericTestData=cbind(numericTestData,testData[,'y'])
  names(lrnumericTrainData)[16]='y'
  names(lrnumericTestData)[16]='y'
  
  # Train the data before processing the imbalance:
  lrmodel1=glm(y~.,data=lrnumericTrainData,family=binomial)
  summary(lrmodel1)
  
  lrpred1=predict(lrmodel1,lrnumericTestData,type='response')
  glm.pred1=rep('no',nrow(lrnumericTestData))
  glm.pred1[lrpred1>0.5]='yes'
  glm.pred1
  table(testData$y,glm.pred1)
  mcc_lr1_before[i]=MCC(testData$y,glm.pred1)
  F_lr1_before[i]=F_measure(testData$y,glm.pred1)
  a_lr1_before[i]=accuracy(testData$y,glm.pred1)
  
  # Plot ROC curve 
  pred_roc = prediction(lrpred1, as.numeric(testData$y=="yes"))
  perf_roc <- performance(pred_roc,"tpr","fpr")
  plot(perf_roc,colorize=TRUE, main = "ROC Curve of Logistic Regression Model using all the features (unbalanced data)")
  abline(a = 0, b = 1)
  # Get AUC value
  auc_roc <- performance(pred_roc, measure = "auc")@y.values[[1]]
  auc_lr1_before[i]=auc_roc
  
  # Train the data after processing the imbalance:
  After_trainData = ovun.sample(y ~ ., data = trainData, method = "both", p=0.5,N=36200, seed = 123)$data
  numericTrainData_after= data.frame(sapply(After_trainData[,-16],as.numeric))
  lrnumericTrainData_after=cbind(numericTrainData_after,After_trainData[,'y'])
  names(lrnumericTrainData_after)[16]='y'
  
  lrmodel2=glm(y~.,data=lrnumericTrainData_after,family=binomial)
  summary(lrmodel2)
  
  lrpred2=predict(lrmodel2,lrnumericTestData,type='response')
  glm.pred2=rep('no',nrow(lrnumericTestData))
  glm.pred2[lrpred2>0.5]='yes'
  glm.pred2
  table(testData$y,glm.pred2)
  mcc_lr1_after[i]=MCC(testData$y,glm.pred2)
  F_lr1_after[i]=F_measure(testData$y,glm.pred2)
  a_lr1_after[i]=accuracy(testData$y,glm.pred2)
  
  # Plot ROC curve 
  pred_roc = prediction(lrpred2, as.numeric(testData$y=="yes"))
  perf_roc <- performance(pred_roc,"tpr","fpr")
  plot(perf_roc,colorize=TRUE, main = "ROC Curve of Logistic Regression Model Using All the Features (balanced data)")
  abline(a = 0, b = 1)
  # Get AUC value
  auc_roc <- performance(pred_roc, measure = "auc")@y.values[[1]]
  auc_lr1_after[i]=auc_roc
}
sum(mcc_lr1_before)/5 
sum(mcc_lr1_after)/5
sum(F_lr1_before)/5
sum(F_lr1_after)/5
sum(a_lr1_before)/5
sum(a_lr1_after)/5
sum(auc_lr1_before)/5
sum(auc_lr1_after)/5
# Since the outcome becomes better after dealing with the imbalance,
# we continue with normalization upon the data set after dealing with the imbalance:

mcc_lr2=rep(0,5)
F_lr2=rep(0,5)
a_lr2=rep(0,5)
auc_lr2=rep(0,5)
for(i in 1:5)
{
  testIndexes = which (folders==i)
  testData = shuffle_data[testIndexes,]
  trainData = shuffle_data[-testIndexes,]
  table(testData$y)
  
  # Train the data after processing the imbalance and normalization:
  After_trainData = ovun.sample(y ~ ., data = trainData, method = "both", p=0.5,N=36200, seed = 123)$data
  numericTrainData_after= data.frame(sapply(After_trainData[,-16],as.numeric))
  numericTestData= data.frame(sapply(testData[,-16],as.numeric))
  
  # Scale data for linear regression:
  lrTrainData = data.frame(lapply(numericTrainData_after, normalize))
  lrTestData = data.frame(lapply(numericTestData, normalize))
  
  LRTrainData = cbind(lrTrainData, After_trainData[,16])
  LRTestData = cbind(lrTestData,testData[,16])
  
  names(LRTrainData)[16]='y'
  names(LRTestData)[16]='y'
  
  
  lrmodel3=glm(y~.,data=LRTrainData,family=binomial)
  summary(lrmodel3)
  
  lrpred3=predict(lrmodel3,LRTestData,type='response')
  glm.pred3=rep('no',nrow(LRTestData))
  glm.pred3[lrpred3>0.5]='yes'
  glm.pred3
  table(testData$y,glm.pred3)
  mcc_lr2[i]=MCC(testData$y,glm.pred3)
  F_lr2[i]=F_measure(testData$y,glm.pred3)
  a_lr2[i]=accuracy(testData$y,glm.pred3)
  
  # Plot ROC curve 
  pred_roc = prediction(lrpred3, as.numeric(testData$y=="yes"))
  perf_roc <- performance(pred_roc,"tpr","fpr")
  plot(perf_roc,colorize=TRUE, main = "ROC Curve of Logistic Regression Model after Normalization (balanced data)")
  abline(a = 0, b = 1)
  # Get AUC value
  auc_roc <- performance(pred_roc, measure = "auc")@y.values[[1]]
  auc_lr2[i]=auc_roc
}
sum(mcc_lr2)/5 
sum(F_lr2)/5
sum(a_lr2)/5
sum(auc_lr2)/5
# Since both MCC and F-measure become worse after normalization, therefore we ignore
# this method and perform feature selection:
summary(lrmodel2)

# After checking the p-value for each attribute, we delete those whose p-value is not
# significantly small like 'job', 'day' and 'default':
# subset selection 1:
mcc_lr3=rep(0,5)
F_lr3=rep(0,5)
a_lr3=rep(0,5)
auc_lr3=rep(0,5)
for(i in 1:5)
{
  testIndexes = which (folders==i)
  testData = shuffle_data[testIndexes,]
  trainData = shuffle_data[-testIndexes,]
  table(testData$y)
  
  ## convert factor variable into numerical type  
  After_trainData = ovun.sample(y ~ ., data = trainData, method = "both", p=0.5,N=36200, seed = 123)$data
  numericTrainData_after= data.frame(sapply(After_trainData[,-16],as.numeric))
  lrnumericTrainData_after=cbind(numericTrainData_after,After_trainData[,'y'])
  names(lrnumericTrainData_after)[16]='y'
  
  numericTestData= data.frame(sapply(testData[,-16],as.numeric))
  lrnumericTestData=cbind(numericTestData,testData[,'y'])
  names(lrnumericTestData)[16]='y'
  
  
  lrmodel4=glm(y~age+marital+education+balance+housing+loan+contact+month+duration+campaign+pdays+previous,data=lrnumericTrainData_after,family=binomial)
  summary(lrmodel4)
  
  lrpred4=predict(lrmodel4,lrnumericTestData,type='response')
  glm.pred4=rep('no',nrow(lrnumericTestData))
  glm.pred4[lrpred4>0.5]='yes'
  glm.pred4
  table(testData$y,glm.pred4)
  mcc_lr3[i]=MCC(testData$y,glm.pred4)
  F_lr3[i]=F_measure(testData$y,glm.pred4)
  a_lr3[i]=accuracy(testData$y,glm.pred4)
  
  # Plot ROC curve 
  pred_roc = prediction(lrpred4, as.numeric(testData$y=="yes"))
  perf_roc <- performance(pred_roc,"tpr","fpr")
  plot(perf_roc,colorize=TRUE, main = "ROC Curve of Logistic Regression Model after Feature Selection (balanced data)")
  abline(a = 0, b = 1)
  # Get AUC value
  auc_roc <- performance(pred_roc, measure = "auc")@y.values[[1]]
  auc_lr3[i]=auc_roc
}
sum(mcc_lr3)/5 
sum(F_lr3)/5
sum(a_lr3)/5
sum(auc_lr3)/5

# Bayesian Classification:
library(e1071)
mcc_bc_before=rep(0,5)
mcc_bc_after=rep(0,5)
F_bc_before=rep(0,5)
F_bc_after=rep(0,5)
a_bc_before=rep(0,5)
a_bc_after=rep(0,5)
auc_bc_before=rep(0,5)
auc_bc_after=rep(0,5)
for(i in 1:5)
{
  testIndexes = which (folders==i)
  testData = shuffle_data[testIndexes,]
  trainData = shuffle_data[-testIndexes,]
  table(testData$y)
  
  # train the model before processing the imbalance:
  Naive_Bayes_Model_before=naiveBayes(y ~., data=trainData)
  pred11=predict(Naive_Bayes_Model_before,testData)
  table(testData$y,pred11)
  mcc_bc_before[i]=MCC(testData$y,pred11)
  F_bc_before[i]=F_measure(testData$y,pred11)
  a_bc_before[i]=accuracy(testData$y,pred11)
  
  # Plot ROC curve 
  library("ROCR")
  pred_roc = prediction(predict(Naive_Bayes_Model_before,testData,type="raw")[,2], testData$y)
  perf_roc <- performance(pred_roc,"tpr","fpr")
  plot(perf_roc,colorize=TRUE, main = "ROC Curve of Naive Bayes Model (unbalanced data)")
  abline(a = 0, b = 1)
  # Get AUC value
  auc_roc <- performance(pred_roc, measure = "auc")@y.values[[1]]
  auc_bc_before[i]=auc_roc
  
  # train the model after processing the imbalance:
  After_trainData = ovun.sample(y ~ ., data = trainData, method = "both", p=0.5,N=36200, seed = 123)$data
  Naive_Bayes_Model_after=naiveBayes(y ~., data=After_trainData)
  pred12=predict(Naive_Bayes_Model_after,testData)
  table(testData$y,pred12)
  mcc_bc_after[i]=MCC(testData$y,pred12)
  F_bc_after[i]=F_measure(testData$y,pred12)
  a_bc_after[i]=accuracy(testData$y,pred12)
  
  # Plot ROC curve 
  library("ROCR")
  pred_roc = prediction(predict(Naive_Bayes_Model_after,testData,type="raw")[,2], testData$y)
  perf_roc <- performance(pred_roc,"tpr","fpr")
  plot(perf_roc,colorize=TRUE, main = "ROC Curve of Naive Bayes Model(balanced data)")
  abline(a = 0, b = 1)
  # Get AUC value
  auc_roc <- performance(pred_roc, measure = "auc")@y.values[[1]]
  auc_bc_after[i]=auc_roc
}
sum(mcc_bc_before)/5
sum(mcc_bc_after)/5
sum(F_bc_before)/5
sum(F_bc_after)/5
sum(a_bc_before)/5
sum(a_bc_after)/5
sum(auc_bc_before)/5
sum(auc_bc_after)/5





# Neural Network:
library(neuralnet)
mcc_nn_before=rep(0,5)
mcc_nn_after=rep(0,5)
F_nn_before=rep(0,5)
F_nn_after=rep(0,5)
a_nn_before=rep(0,5)
a_nn_after=rep(0,5)
for(i in 1:5){
  testIndexes = which (folders==i)
  testData = shuffle_data[testIndexes,]
  trainData = shuffle_data[-testIndexes,]
  table(testData$y)
  
  # convert factor variable into numerical type  
  numericTrainData= data.frame(sapply(trainData[,-16],as.numeric))
  numericTestData= data.frame(sapply(testData[,-16],as.numeric))
  
  # Scale data for neural network:
  nnTrainData = data.frame(lapply(numericTrainData, normalize))
  nnTestData = data.frame(lapply(numericTestData, normalize))
  
  NNTrainData = cbind(nnTrainData, trainData[,16])
  NNTestData = cbind(nnTestData,testData[,16])
  names(NNTrainData)[16]='y'
  names(NNTestData)[16]='y'

  n1 = names(NNTrainData)
  f1 = as.formula(paste('y~',paste(n1[!n1 %in% 'y'], collapse='+')))
  nn1 = neuralnet(f1,data=NNTrainData,hidden=3,linear.output = F,stepmax = 1e6)
  plot(nn1)
  prednn1=compute(nn1,NNTestData[,1:16])
  prednn1
  nn.pred1=rep('no',nrow(NNTestData))
  prednn1$net.result[,1]
  nn.pred1[prednn1$net.result[,1]>0.5]='yes'
  table(testData$y,nn.pred1)
  mcc_nn_before[i]=MCC(testData$y,nn.pred1)
  F_nn_before[i]=F_measure(testData$y,nn.pred1)
  a_nn_before[i]=accuracy(testData$y,nn.pred1)
  
  # After processing the imbalance data:
  After_trainData = ovun.sample(y ~ ., data = trainData, method = "both", p=0.5,N=36200, seed = 123)$data
  numericTrainData_after= data.frame(sapply(After_trainData[,-16],as.numeric))
  numericTestData= data.frame(sapply(testData[,-16],as.numeric))
  
  # Scale data for linear regression:
  nnTrainData_after = data.frame(lapply(numericTrainData_after, normalize))
  nnTestData_after = data.frame(lapply(numericTestData, normalize))
  
  NNTrainData_after = cbind(nnTrainData_after, After_trainData[,16])
  NNTestData_after = cbind(nnTestData_after,testData[,16])
  
  names(NNTrainData_after)[16]='y'
  names(NNTestData_after)[16]='y'

  n2 = names(NNTrainData_after)
  f2 = as.formula(paste('y~',paste(n2[!n2 %in% 'y'], collapse='+')))
  nn2 = neuralnet(f2,data=NNTrainData_after,hidden=3,linear.output = F,stepmax = 1e6)
  plot(nn2)
  prednn2=compute(nn2,NNTestData_after[,1:16])
  prednn2
  nn.pred2=rep('no',nrow(NNTestData_after))
  prednn2$net.result[,1]
  nn.pred2[prednn2$net.result[,1]>0.5]='yes'
  table(testData$y,nn.pred2)
  mcc_nn_after[i]=MCC(testData$y,nn.pred2)
  F_nn_after[i]=F_measure(testData$y,nn.pred2)
  a_nn_after[i]=accuracy(testData$y,nn.pred2)
  
}

sum(mcc_nn_before)/5
sum(F_nn_before)/5
sum(a_nn_before)/5
sum(mcc_nn_after)/5
sum(F_nn_after)/5
sum(a_nn_after)/5

























