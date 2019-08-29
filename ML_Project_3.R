################################################################
# Ramesh Subedi

# Use the following two learning algorithms:
# 1. Artificial neural networks (ANN)
# 2. K Nearest Neighbors
# 
# Tasks:
# 1. Download and use any neural networks package to classify your classification problems. 
# Experiment with number of layers and number of nodes, activation functions (sigmoid, tanh, etc.), 
# and may be a couple of other parameters.
# 
# 2. Download and use any KNN package to classify your classification problems. 
# Experiment with number of neighbors. You can use any distance metric appropriate 
# to your problem. Just be clear to explain why you used the metric that you used.
# 
# Include these:
# •Error rates (train and test) for the two algorithms on your two data sets. 
# Plot various types of learning curves that you can think of 
# (e.g. but not limited to – error rates vs. train data size, error rates vs. clock time 
#   to train/test, etc.).
# • Performance comparisons (learning curves, confusion matrices, etc.) 
# of various functions/parameters for the algorithms 
# (e.g. ANN number of layers, nodes, etc.,number of neighbors in KNN, etc.) on both the data sets.
# • Comparisons of the two learning algorithms using the two data sets.

###############################################################

######################################
library(data.table) 
library(magrittr)
library(plyr) 
library(dtplyr) 
library(sandwich) # for White correction
#library(lmtest) # for more advanced hypothesis testing tools
#library(tseries) # time series package
#library(DBI) 
#library(RSQLite) 
library(tidyverse)
library(broom)  # for tidy() function
#library(TSA)
#library(forecast)
library(vars)
#library(fpp) # for VAR forecast
library(UsingR)
#library(margins)
#library(plm) # for pooled OLS (ordinary least squares)
library(car) # for scatterplot()
#library(aod) # for probit link
library(gradDescent) # for Gradient Descent calculation
#library(glmnet)
library(e1071) # for Support Vector Machine, Titanic data, etc.
library(tree) # for tree to work on Decisiion Trees
library(gbm) # for gbm (gradient boosting model)
library(adabag) # for bagging
library(rpart) # 
library(party) # Recursive partitioning
library(partykit) # Pruning tree from party
library(neuralnet) # for neural net
library(caret) # for KNN
library(ROCR) # for KNN as well
library(pROC) # for KNN as well
library(boot) # for cross-validation
library(ggplot2)
library(class)

rm(list=ls()) #drop all variables

data <- read.table("~/mlData/student-mat.csv",sep=";",header=TRUE)


set.seed(501)
data <- data %>% dplyr::select(-G1,-G2) # Drop G2 and G2 variables from data
names(data)
data$school <- as.numeric(data$school)
data$school
str(data$school)

data$sex <- as.numeric(data$sex)
data$address<-as.numeric(data$address)
data$famsize <- as.numeric(data$famsize)
data$Pstatus<- as.numeric(data$Pstatus)
data$Fjob<-as.numeric(data$Fjob)
data$Mjob<- as.numeric(data$Mjob)
data$reason<- as.numeric(data$reason)
data$guardian<- as.numeric(data$guardian)
data$schoolsup<- as.numeric(data$schoolsup)
data$famsup<- as.numeric(data$famsup)
data$paid<- as.numeric(data$paid)
data$activities<- as.numeric(data$activities)
data$nursery<- as.numeric(data$nursery)
data$higher<- as.numeric(data$higher)
data$internet<- as.numeric(data$internet)
data$romantic<- as.numeric(data$romantic)

data%>%mutate_if(is.numeric, scale) # This scaling works. Though scaling is only for numeric variables (not for factors), we changed factors into numeric above. Hence this scaling works for all variables.

names(data)
class(data) # To check if data is still data.frame


# make G3 in data a binary varialbe (0 or 1)  since GLM needs this.
Grade <- data$G3
Grade
MeanVal<- mean(data$G3)
MeanVal
MyGrade <- ifelse(Grade>=MeanVal,1,0)
MyGrade

# Add variable MyGrade to existing data
newData <- cbind(data,MyGrade)
# remove (drop) G3 from newData:
newData <- newData %>% dplyr::select(-G3)
data <- newData
class(data)
names(data)


#######################################

set.seed(500)
# Check that no data is missing
apply(data,2,function(x) sum(is.na(x)))  # zeros for all variables, no missing data

# Train-test random splitting for linear model
index.data <- sample(1:nrow(data),round(0.7*nrow(data)))
Train.data <- data[index.data,] # 70%
Test.data <- data[-index.data,] # 30%

dim(Train.data) #  276  31
dim(Test.data) # 119  31

sum(is.na(Train.data)) # 0
sum(is.na(Test.data)) # 0
names(Train.data)

# Fitting generalized linear model
linMod.fit <- glm(MyGrade~., data=Train.data)
summary(linMod.fit)

# Predicted data from lm
pred.linMod <- predict(linMod.fit,Test.data)
pred.linMod


# Test MSE
MSE.linMod <- sum((pred.linMod - Test.data$MyGrade)^2)/nrow(Test.data)
MSE.linMod  # 0.235913649509655

names(data)

# Scaling data for the Neural Net
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
names(scaled)
# Checking if data has missing values (NA)
#is.na(x) # returns TRUE of x is missing
sum(is.na(scaled)) # 0 if no NA, non-zero if NA's are present
sum(is.na(scaled))>0 # FALSE means no NA present

str(scaled)


# Check that no data is missing
apply(scaled,2,function(x) sum(is.na(x))) # good, zeros for all variables, no NA's found


# Train-test split
trainScaled <- scaled[index.data,]
testScaled <- scaled[-index.data,]

names(trainScaled)
dim(trainScaled) #   276  31
sum(is.na(trainScaled)) # 0, good
sum(is.na(testScaled)) # 0, good
dim(testScaled) # 119  31

str(trainScaled)
str(testScaled)
names(testScaled)

NN <- names(trainScaled)
ff <- as.formula(paste("MyGrade ~", paste(NN[!NN %in% "MyGrade"], collapse = " + ")))
ff

# For learning curve, run the neural network for multiple times by changing the percentage of 
# train/test split and find SSE for each step. Plot that sse in the graph.
# Calculating Sum of Squared Error (SSE or RSS (residual sum of squared)) and MSE.

NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Train_SSE <- sum((NeuralNet$net.result[[1]] - trainScaled[,31])^2)
MSE_train = NN_Train_SSE/nrow(trainScaled)
MSE_train # Mean Squared Error  0.07495922669
stdTr = sqrt(NN_Train_SSE/nrow(trainScaled)) 
stdTr   # 0.2737868271
nrow(trainScaled) # 276
# 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45
# stdTr: 0.2310938442, 0.2465022793, 0.2737868271, 0.2720719734, 0.1688905424, 0.2311349103, 0.2494427182, 0.1672886349
listSE = c(0.2310938442, 0.2465022793, 0.2737868271, 0.2720719734, 0.1688905424, 0.2311349103, 0.2494427182) # for train data

listMSE = c(0.2310938442^2, 0.2465022793^2, 0.2737868271^2, 0.2720719734^2, 0.1688905424^2, 0.2311349103^2, 0.2494427182^2) # for train data

NeuralNetTest <- neuralnet(ff,data=testScaled,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Test_SSE <- sum((NeuralNetTest$net.result[[1]]-testScaled[,31])^2)
stdTe = sqrt(NN_Test_SSE/nrow(testScaled)) 
stdTe
nrow(testScaled)
# stdTe: 0.1388068321, 0.1569529771, 0.1555027501, 0.1447596063, 0.1448921464, 0.1540366963, 0.1235372153,  0.3202106573
listSE1 = c(0.1388068321, 0.1569529771, 0.1555027501, 0.1447596063, 0.1448921464, 0.1540366963, 0.1235372153) # for test data

listMSE1 = c(0.1388068321^2, 0.1569529771^2, 0.1555027501^2, 0.1447596063^2, 0.1448921464^2, 0.1540366963^2, 0.1235372153^2) # for test data

# SSE for test data: 1.522119594 2.438789466 2.877551531  2.891837418 3.317009986  4.223460079   3.006504384 22.25006572


# nrow(trainScaaled): 316, 296, 276, 257, 237, 217, 198, 178
# nrow(testScaaled): 79, 99, 119, 138, 158, 178,197,217 

list2b=c(316, 296, 276, 257, 237, 217, 198)
#list1 = c(16.87577929,  17.98595862, 20.68874657, 19.02395179 , 6.760191632, 11.59286624, 12.31989059)
#list2 = c(0.8,  0.75,  0.7,  0.65,  0.6 ,  0.55,   0.5, 0.45) # these are % split of train/test

list2a = c(79, 99, 119, 138, 158, 178,197)
#list3 = c(1.522119594, 2.438789466, 2.877551531,  2.891837418, 3.317009986, 4.223460079,   3.006504384)

par(mfrow=c(2,1))
plot (listMSE1~list2a, type = "l", xlab = "Test data size", ylab = "Test Data MSE", main = "Learning curve for test dataset (School Data)")
plot (listMSE~list2b, type = "l", xlab = "Training data size", ylab = "Training Data MSE", main = "Learning curve for training dataset  (School Data)")
par(mfrow=c(1,1))

# Neural net with two hidden layers c(5,3): first with 5 nodes and second with 3 nodes

# The activation function used in both neuralnet or nnet is a sigmoid (logistic) function by default. 

# Maximum 3 hidden layers are enough for any complex problems.
# Default threshold=0.01. Change it to threshold=0.05 so that there is no convergence issue
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(3,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(4,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(4,3), threshold = 0.05, act.fct='tanh') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,3), threshold = 0.05, act.fct='tanh') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,4), threshold = 0.05, act.fct='tanh') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(4,3,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,3,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,4,2), threshold = 0.05, act.fct='tanh')
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,4,3), threshold = 0.05, act.fct='tanh') 


# With rep=10 we construct 10 different ANNs, and select the best of the 10.
# NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(3,2), threshold = 0.05, rep=10,act.fct='logistic')
NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(4,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(4,3), threshold = 0.05, act.fct='logistic') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,3), threshold = 0.05, act.fct='logistic') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,4), threshold = 0.05, act.fct='logistic') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(4,3,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,3,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,4,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(5,4,3),threshold = 0.05, act.fct='logistic', rep=10) 

length(NeuralNet) # 13
NeuralNet
NeuralNet$call
NeuralNet$net.result
names(NeuralNet)
NeuralNet$act.fct # logistic whether linear.output=T or F, tanh only for linear.output=F
NeuralNet$weights
NeuralNet$result.matrix
plot(NeuralNet$response)
NeuralNet$model.list

 #plot(NeuralNet) # plots out of RStudio window
 plot(NeuralNet, rep="best") # plots in RStudio window
# https://rpubs.com/Spencer_Butt_RPubs/301131
 
# Predict

pred.NeuralNet <- compute(NeuralNet,testScaled[,1:30])
length(pred.NeuralNet)
prob.result <- pred.NeuralNet$net.result
####### for ROC curve #######################

nn.pred = prediction(prob.result, testScaled$MyGrade)
pref <- performance(nn.pred, "tpr", "fpr")
plot(pref,col = "blue", lwd = 1.5,main="School Data ROC from ANN", xlab='False positive rate (or 1-specificity)',ylab='True positive rate (or sensitivity)')
#Calculating KS statistics (KS for Kolmogorov-Smirnov)
ksANN <- max(attr(pref, "y.values")[[1]] - (attr(pref, "x.values")[[1]]))
ksANN  #  0.3517260894
# Calculate the Area under curve (auc) on test dataset
attributes(performance(nn.pred, 'auc'))$y.values[[1]]  
# Area =  0.6813808715 


trainScaled$MyGrade
plot(trainScaled[,31])
plot(NeuralNet$net.result[[1]]) # this is how you plot a list object.

class(NeuralNet$net.result)
class(pred.NeuralNet)
pred.NeuralNet
pred.NeuralNet[[2]]

########## ROC curve finished #################

# Results from Neural Net are normalized (scaled).
# Descaling for comparison
pred.NeuralNet_ <- pred.NeuralNet$net.result*(max(MyGrade)-min(MyGrade))+min(MyGrade)
length(pred.NeuralNet_)
myTest <- (testScaled$MyGrade)*(max(MyGrade)-min(MyGrade))+min(MyGrade)
length(testScaled$MyGrade)
length(max(MyGrade)-min(MyGrade))
length(MyGrade)
length(myTest)
# Calculating MSE
MSE.NeuralNet <- sum((myTest - pred.NeuralNet_)^2)/nrow(testScaled)
MSE.NeuralNet 
# Compare the two MSEs
print(paste(MSE.linMod,MSE.NeuralNet)) 


# To get the same value in different try, you need to repeat the operation all over again (start from 
# rm(list=ls()).

# for tanh
# for c(3,2):    0.243493045953089 0.376858284191622
# for c(4,2):    0.243493045953089 0.847373410805842 
# for c(4,3):    0.243493045953089 0.485146931070311 
# for c(5,2):    0.243493045953089 0.330016785536778
# for c(5,3):    0.243493045953089 0.487771593316403
# for c(5,4):    0.243493045953089 0.54358335318412 
# for c(4,3,2):  0.243493045953089 0.344313671167737
# for c(5,3,2):  0.243493045953089 0.419546543285571 
# for c(5,4,2):  0.243493045953089 0.378202961840539
# for c(5,4,3):  0.243493045953089 0.418506051289785

# for logistic
# for c(3,2):   0.243493045953089 0.288905838391712
# for c(4,2): 
# for c(4,3): 
# for c(5,2):   0.243493045953089 0.354442476440748
# for c(5,3):   0.243493045953089 0.394818373049363
# for c(5,4): 
# for c(4,3,2):
# for c(5,3,2):
# for c(5,4,2): 
# for c(5,4,3): 0.243493045953089 0.482020099339624

A <- c(0.531631751865308, 0.312796578222351,  0.264539724644674) # MSE from NN tanh
B <- c(0.3548736324667, 0.419503912236768, 0.480041687937691) # MSE from NN logistic
C <- c(0.235913649509655) # MSE from GLM
hist(A, col=rgb(0.2,0.8,1/4),xlim=c(0.0,0.6), ylim=c(0.15,2.), main="NN tanh (green), NN logistic (pink) \n NN tanh and NN logistic overlap (dark green) \n Tanh, logistic, GLM overlap (coffee color)",xlab="MSE with ANN analysis")
#hist(B, col=rgb(1,0,0,1/4), add=T)
hist(B, col=rgb(1,0,0.4,1/4), add=T)
hist(C, col=rgb(0.2,0,0,1/2), add=T)
box()


# Plot predictions
par(mfrow=c(3,1))
plot(Test.data$MyGrade,pred.NeuralNet_,col='red',main='Real vs NN predicttion',pch=18,cex=0.97,xlab='MyGrade', ylab='NN prediction')
#plot(Test.data$G3,pred.NeuralNet_,col='red',main='Real vs NN prediction',pch=18,cex=0.7)
#abline(0,1,lwd=2)
legend('top',legend='NN',pch=18,col='red', bty='n')

plot(Test.data$MyGrade,pred.linMod,col='blue',main='Real vs GLM prediction',pch=18, cex=0.97,xlab='MyGrade', ylab='GLM prediction')
#plot(Test.data$G3,pred.linMod,col='blue',main='Real vs GLM prediction',pch=18, cex=0.7)
#abline(0,1,lwd=2)
legend('top',legend='GLM',pch=18,col='blue', bty='n', cex=.95)
#par(mfrow=c(1,1))
# Compare predictions on the same plot
plot(Test.data$MyGrade,pred.NeuralNet_,col='red',main='Above two plots in the same canvas: Real vs predicted MyGrade',pch=18,cex=0.7,xlab='MyGrade',ylab='Predicted MyGrade')
points(Test.data$MyGrade,pred.linMod,col='blue',pch=18,cex=0.97)
#abline(0,1,lwd=2)
legend('top',legend=c('NN','GLM'),pch=18,col=c('red','blue'))
# The linear model has lesser spread than the artificial neural network (ANN) model.
par(mfrow=c(1,1))
#-------------------------------------------------------------------------------
# Cross validating

# library(boot)
#set.seed(201)

# Linear model cross validation
lm.Fit <- glm(MyGrade~.,data=data)
length(lm.Fit)
length(data)
set.seed(201)
cv.glm(data,lm.Fit,K=10)
names(cv.glm(data,lm.Fit,K=10))
cv.glm(data,lm.Fit,K=10)$delta
cv.glm(data,lm.Fit,K=10)$delta[1] # 0.2357927667

names(data)

# Neural net cross validation
set.seed(455)
cv.error <- NULL
sseTrain <-NULL
sseTest <-NULL
k <- 10  # k=10 means 10-fold cross validation, can do for k=100 : it's 100 samples

# Initialize progress bar
# library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  
  ann <- neuralnet(ff,data=trainScaled,hidden=c(5,2),threshold = 0.05) 
  length(ann)
  pr.ann <- compute(ann,testScaled[,1:30])
  length(pr.ann)
  prob.result <- pr.ann$net.result
  
  pr.ann1 <- pr.ann$net.result*(max(MyGrade)-min(MyGrade))+min(MyGrade)
  length(pr.ann1)
  test.cv.r <- (testScaled$MyGrade)*(max(MyGrade)-min(MyGrade))+min(MyGrade)
  length(test.cv.r)
  length(testScaled)
  
  pred.annTrain <- compute(ann,trainScaled[,1:30])
  sseTrain[i] <- sum((pred.annTrain$net.result[[1]] - trainScaled[,31])^2) # SSE for train data
  sseTest[i] <- sum((pr.ann[[2]] - testScaled[,31])^2) # SSE for test data
  
  cv.error[i] <- sum((test.cv.r - pr.ann1)^2)/nrow(testScaled)
  pbar$step()
}
pred.annTrain$net.result[[2]] 
pr.ann[[2]]
# Average MSE
mean(cv.error) # 0.4431545594
median(cv.error) # 0.4122028095
# MSE vector from CV
cv.error
sseTrain
sseTest
mean(sseTrain)
mean(sseTest)

par(mfrow=c(3,1))
# Visual plot of CV results
boxplot(cv.error,xlab='MSE with Cross-Validation',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN for School Data',horizontal=TRUE) # to shrink horizontally, change ylim (not xlim), this box plot has one outlier in the right side of the whisker little beyound 1.2.


# Visual plot of sseTrain results
boxplot(sseTrain,xlab='SSE of Training Data',col='cyan',
        border='blue', main='CV error (SSE) for NN for School Training Data',horizontal=TRUE) 

# Visual plot of sseTest results
boxplot(sseTest,xlab='SSE of Test Data',col='cyan',
        border='blue', main='CV error (SSE) for NN for School Test Data',horizontal=TRUE) 
par(mfrow=c(1,1))

# Plots for report
par(mfrow=c(2,1))
hist(A, col=rgb(0.2,0.8,1/4),xlim=c(0.15,0.6), ylim=c(0.15,2.), main="NN tanh (green), NN logistic (pink) \n NN tanh and NN logistic overlap (dark green) \n Tanh, logistic, GLM overlap (coffee color)",xlab="MSE with ANN analysis")
#hist(B, col=rgb(1,0,0,1/4), add=T)
hist(B, col=rgb(1,0,0.4,1/4), add=T)
hist(C, col=rgb(0.2,0,0,1/2), add=T)
box()

boxplot(cv.error,xlab='MSE with Cross-Validation',col='cyan',
        border='blue',names='CV error (MSE)',
        main='Cross-Validation error (MSE) for NN',horizontal=TRUE,ylim=c(0.15,0.6))
par(mfrow=c(1,1))


######################## End of ANN of Data1 #####################
####################### Start of KNN of Data1 ####################

set.seed(101)

# Explore data
dim(trainScaled)
dim(testScaled)
names(trainScaled)
head(trainScaled)
head(testScaled)
trainScaled$MyGrade

trainScaled$MyGrade=ifelse(trainScaled$MyGrade==0,"X0","X1") # change the levels to X0 and X1 from 0 and 1
testScaled$MyGrade=ifelse(testScaled$MyGrade==0,"X0","X1")
trainScaled$MyGrade
testScaled$MyGrade

# Setting levels for both training and validation data
levelsSchool_trainScaled<- make.names(levels(factor(trainScaled$MyGrade)))
levelsSchool_testScaled <- make.names(levels(factor(testScaled$MyGrade)))
levelsSchool_trainScaled
levelsSchool_testScaled

# Setting up train controls
repeats = 3
numbers = 10
tunel = 20   #10
numb = 60
set.seed(1234)

xSchool = trainControl(method = "repeatedcv",
                 number = numbers,
                 repeats = repeats,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)

modelSchool <- train(MyGrade~. , data = trainScaled, method = "knn",
                preProcess = c("center","scale"),tuneGrid = expand.grid(k = 1:numb),
                trControl = xSchool,
                metric = "ROC", 
                tuneLength = tunel)

summary(modelSchool)
modelSchool
plot(modelSchool,main='School Data')

# Testing
test_predSchool <- predict(modelSchool,testScaled, type = "prob") 
test_predSchool
test_predSchool_1 <- predict(modelSchool,testScaled)
test_predSchool_1

confusionMatrix(test_predSchool_1, as.factor(testScaled$MyGrade))

# Confusion Matrix and Statistics
#            Reference
# Prediction  0  1
#          0 29 24
#          1 16 50
# 
# Accuracy : 0.6639          
# 95% CI : (0.5715, 0.7478)

#Storing Model Performance Scores
testScaled$MyGrade
testScaled$MyGrade2=ifelse(testScaled$MyGrade=="X0",0,1) # change the levels back to 0 and 1
pred_valSchool <-prediction(test_predSchool[,2],testScaled$MyGrade2)
testScaled$MyGrade2

# Calculating Area under Curve (AUC)
perf_valSchool <- performance(pred_valSchool,"auc")
perf_valSchool

# Plot AUC
perf_valSchool <- performance(pred_valSchool, "tpr", "fpr")
plot(perf_valSchool, col = "blue", lwd = 1.5,main="School Data ROC from KNN",xlab='False positive rate (or 1-specificity)',ylab='True positive rate (or sensitivity')

#Calculating KS statistics (KS for Kolmogorov-Smirnov)
ksSchool <- max(attr(perf_valSchool, "y.values")[[1]] - (attr(perf_valSchool, "x.values")[[1]]))
ksSchool  #  0.3005093379 

# Calculate the Area under curve (AUC) on validation dataset
attributes(performance(pred_valSchool, 'auc'))$y.values[[1]]  
# Area =  0.6703452179 



######################################################################
################ End of first dataset student-mat.csv #################
######################################################################
## Now we work everything as above with the second data set bank.df ##
######################################################################


###################### Start of ANN for second data set bank.df ######

# Dataset from here:
# https://github.com/gchoi/Dataset/blob/master/UniversalBank.csv

Bank.df <- read.csv("~/mlData/bank.csv", header = TRUE) 
names(Bank.df)
str(Bank.df)
Bank.df$PersonalLoan # already a binary variable
Bank.df$CDAccount
dim(Bank.df) # 5000   14

bank.df <- Bank.df[ , -c(1, 5)]  # Drop ID and zip code columns.
names(bank.df)
dim(bank.df) # 5000   12
bank.df <- bank.df %>% dplyr::select(Age,Experience,Income,Family,CCAvg,Education,Mortgage,SecuritiesAccount,CDAccount,Online,CreditCard,PersonalLoan)  # Make sure PersonalLoan is at the last position.

names(bank.df)
summary(bank.df)

# Check if there is missing data.
apply(bank.df,2,function(x) sum(is.na(x))) # all zeros, good.



set.seed(502)
# partition the data where 70% is train.df and rest is test.df.

index.data1 <- sample(1:nrow(bank.df),round(0.45*nrow(bank.df)))
Train.data1 <- bank.df[index.data1, ]
Test.data1 <- bank.df[-index.data1, ]
#plot(Train.data1$PersonalLoan) # It's already a 0 and 1 plot.

length(index.data1) # 3500

# Fitting generalized linear model
linMod.fit1 <- glm(PersonalLoan~., data=Train.data1)
summary(linMod.fit1)


# Predicted data from lm
pred.linMod1 <- predict(linMod.fit1,Test.data1)
pred.linMod1

# Test MSE
MSE.linMod1 <- sum((pred.linMod1 - Test.data1$PersonalLoan)^2)/nrow(Test.data1)
MSE.linMod1  # 0.05438100183


# Scaling data for the Neural Net
maxs1 <- apply(bank.df, 2, max) 
maxs1
mins1 <- apply(bank.df, 2, min)
mins1
scaled1 <- as.data.frame(scale(bank.df, center = mins1, scale = maxs1 - mins1))
names(scaled1)
# Checking if data has missing values (NA)
#is.na(x) # returns TRUE of x is missing
sum(is.na(scaled1)) # 0 if no NA, non-zero if NA's are present
sum(is.na(scaled1))>0 # FALSE means no NA present

str(scaled1)


# Check that no data is missing
apply(scaled1,2,function(x) sum(is.na(x))) # good, zeros for all variables, no NA's found


# Train-test split
trainScaled1 <- scaled1[index.data1,]
testScaled1 <- scaled1[-index.data1,]

names(trainScaled1)
dim(trainScaled1) # 276  12
sum(is.na(trainScaled1)) # 0, good
sum(is.na(testScaled1)) # 0, good
dim(testScaled1) # 4724   12

NN1 <- names(trainScaled1)
ff1 <- as.formula(paste("PersonalLoan ~", paste(NN1[!NN1 %in% "PersonalLoan"], collapse = " + ")))
ff1


NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Train_SSE1 <- sum((NeuralNet1$net.result[[1]] - trainScaled1[,12])^2)
NN_Train_SSE1
MSE_train = NN_Train_SSE1/nrow(trainScaled1)
MSE_train # Mean Squared Error
stdTr1 = sqrt(NN_Train_SSE1/nrow(trainScaled1)) # Standard deviation (standard error)
stdTr1
nrow(trainScaled1)
# 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45
# 4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250
# SSE1: 38.82654545, 33.50066166, 34.05861149, 27.1635165, 25.05937858, 33.29131574, 30.4993979, 24.63153281
# stdTr: 0.09852226328, 0.09451724592, 0.09864599259, 0.09142212576, 0.09139543858, 0.1100269649, 0.1104525199, 0.1046295748
list11 = c()
NeuralNetTest1 <- neuralnet(ff1,data=testScaled1,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Test_SSE1 <- sum((NeuralNetTest1$net.result[[1]]-testScaled1[,12])^2)
NN_Test_SSE1
stdTe1 = sqrt(NN_Test_SSE1/nrow(testScaled1)) 
stdTe1
nrow(testScaled1)
# 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750
# Test_SSE1: 4.480996875, 32.04634996, 6.368535685, 8.743226009, 13.44990207, 15.0299738,  23.2448625, 20.71839007
# stdTe1: 0.06694024854, 0.160115833, 0.06515896298, 0.07068330176,  0.08200579878, 0.08173119572, 0.09642585234, 0.08679838723

list11b = c(4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250)
list11a = c(1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750)
list22 = c(0.06694024854, 0.160115833, 0.06515896298, 0.07068330176,  0.08200579878, 0.08173119572, 0.09642585234, 0.08679838723) # test data Standard Error
listMSE2 = c(0.06694024854^2, 0.160115833^2, 0.06515896298^2, 0.07068330176^2,  0.08200579878^2, 0.08173119572^2, 0.09642585234^2, 0.08679838723^2) # test data MSE


list33 = c(0.09852226328, 0.09451724592, 0.09864599259, 0.09142212576, 0.09139543858, 0.1100269649, 0.1104525199, 0.1046295748) # train  data Standard Error

listMSE2a = c(0.09852226328^2, 0.09451724592^2, 0.09864599259^2, 0.09142212576^2, 0.09139543858^2, 0.1100269649^2, 0.1104525199^2, 0.1046295748^2) # train data MSE

par(mfrow=c(2,1))
plot (listMSE2~list11a, type = "l", xlab = "Test data size", ylab = "Test Data MSE", main = "Learning curve for test dataset (Bank Data)")
plot (listMSE2a~list11b, type = "l", xlab = "Training data size", ylab = "Training Data MSE", main = "Learning curve for training dataset  (Bank Data)")
par(mfrow=c(1,1))

#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(3,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(4,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(4,3), threshold = 0.05, act.fct='tanh') # no convergence
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,2), threshold = 0.05, act.fct='tanh')  # no convergence
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,3), threshold = 0.05, act.fct='tanh') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,4), threshold = 0.05, act.fct='tanh') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(4,3,2), threshold = 0.05, act.fct='tanh')
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,3,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,4,2), threshold = 0.05, act.fct='tanh') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,4,3), threshold = 0.05, act.fct='tanh')

#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(3,2), threshold = 0.05, act.fct='logistic') 
NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(4,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(4,3), threshold = 0.05, act.fct='logistic') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,3), threshold = 0.05, act.fct='logistic') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,4), threshold = 0.05, act.fct='logistic') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(4,3,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,3,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,4,2), threshold = 0.05, act.fct='logistic') 
#NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(5,4,3), threshold = 0.05, act.fct='logistic') 
plot(NeuralNet1, rep="best")

# Predict
pred.NeuralNet1 <- compute(NeuralNet1,testScaled1[,1:11])
prob.result1 <- pred.NeuralNet1$net.result
names(NeuralNet1)
NeuralNet1$call
nn.pred1 = prediction(prob.result1, testScaled1$PersonalLoan)

pref1 <- performance(nn.pred1, "tpr", "fpr")
plot(pref1,col = "red", lwd = 1.5,main="Bank Data ROC from ANN",ylab=" True Positive Rate (or Sensitivity)", xlab = "False Positive Rate [or (1-specificity)]")
#Calculating KS statistics (KS for Kolmogorov-Smirnov)
ksANN1 <- max(attr(pref1, "y.values")[[1]] - (attr(pref1, "x.values")[[1]]))
ksANN1  #  0.9209863604
# Calculate the Area under curve (auc) on test dataset
attributes(performance(nn.pred1, 'auc'))$y.values[[1]]  
# Area =  0.9626888122 


# Descaling for comparison
pred.NeuralNet1_ <- pred.NeuralNet1$net.result*(max(bank.df$PersonalLoan)-min(bank.df$PersonalLoan))+min(bank.df$PersonalLoan)
myTest1 <- (testScaled1$PersonalLoan)*(max(bank.df$PersonalLoan)-min(bank.df$PersonalLoan))+min(bank.df$PersonalLoan)
# Calculating MSE
MSE.NeuralNet1 <- sum((myTest1 - pred.NeuralNet1_)^2)/nrow(testScaled1)
#MSE.NeuralNet1 
# Compare the two MSEs
print(paste(MSE.linMod1,MSE.NeuralNet1)) 

length(pred.NeuralNet1_) # 1500
length(Test.data1$PersonalLoan) # 1500

# for tanh
# for c(3,2    ): 0.0543810018267865 0.0148456802979186
# for c(4,2    ): 0.0543810018267865 0.0197337478757271
# for c(4,3    ): no convergence
# for c(5,2    ): no convergence
# for c(5,3    ): no convergence
# for c(5,4    ): no convergence
# for c(4,3,2  ): no convergence
# for c(5,3,2  ): no convergence
# for c(5,4,2  ): no convergence
# for c(5,4,3  ): no convergence


# for logistic
# for c(3,2    ):  0.0543810018267865 0.020011555522887
# for c(4,2    ):  0.0543810018267865 0.0174280619692191
# for c(4,3    ):  # do not try, no convergence for tanh
# for c(5,2    ):  # do not try, no convergence for tanh 
# for c(5,3    ):  # do not try, no convergence for tanh 
# for c(5,4    ):  # do not try, no convergence for tanh
# for c(4,3,2  ):  # do not try, no convergence for tanh 
# for c(5,3,2  ):  # do not try, no convergence for tanh 
# for c(5,4,2  ):  # do not try, no convergence for tanh 
# for c(5,4,3  ):  # do not try, no convergence for tanh


AA <- c(0.0148456802979186, 0.0197337478757271)
BB <- c(0.020011555522887, 0.0174280619692191)
CC <- c(0.020011555522887)

hist(AA, col=rgb(0,0,1,1/4), xlim=c(0.013,0.022),ylim=c(0,1.5), main="ANN tanh (blue)\n ANN logistic (brown)\n ANN logistic and GLM overlap (orange)",xlab="MSE with ANN analysis")
hist(BB, col=rgb(0.5,0,0.3,1/4), add=T)
hist(CC, col=rgb(1,0.5,1/4), add=T)
box()

# Plot predictions
par(mfrow=c(3,1))
plot(Test.data1$PersonalLoan,pred.NeuralNet1_,col='red',main='Real vs ANN prediction',pch=18,cex=0.7,xlab='PersonalLoan',ylab='Predicted PersonalLoan')
abline(0,1,lwd=2)
legend('top',legend='ANN',pch=18,col='red', bty='n')

length(Test.data1$PersonalLoan)
length(pred.NeuralNet1_)

plot(Test.data1$PersonalLoan,pred.linMod1,col='blue',main='Real vs GLM prediction',pch=18, cex=0.7,xlab='PersonalLoan',ylab='Predicted PersonalLoan')
abline(0,1,lwd=2)
legend('top',legend='GLM',pch=18,col='blue', bty='n', cex=.95)

# Compare predictions on the same plot
plot(Test.data1$PersonalLoan,pred.NeuralNet1_,col='red',main='Real vs ANN or GLM Prediction',pch=18,cex=0.7,xlab='PersonalLoan',ylab='Predicted PersonalLoan')
points(Test.data1$PersonalLoan,pred.linMod1,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('top',legend=c('ANN','GLM'),pch=18,col=c('red','blue'))
par(mfrow=c(1,1))

#-------------------------------------------------------------------------------
# Cross validating

# library(boot)
set.seed(202)

# Linear model cross validation
lm.Fit1 <- glm(PersonalLoan~.,data=bank.df)
cv.glm(bank.df,lm.Fit1,K=10)$delta[1] # 0.05361207781


# Neural net cross validation
set.seed(452)
cv.error1 <- NULL
k <- 10  # 10-fold cross validation

# Initialize progress bar
library(plyr) 
pbar1 <- create_progress_bar('text')
pbar1$init(k)

for(i in 1:k){
  index1 <- sample(1:nrow(bank.df),round(0.9*nrow(bank.df)))
  train.cv1 <- scaled1[index1,]
  test.cv1 <- scaled1[-index1,]
  names(train.cv1)
  
  nn1 <- neuralnet(ff1,data=train.cv1,hidden=c(3,2),threshold = 0.05)
  
  pr.nn1 <- compute(nn1,test.cv1[,1:11])
  pr.nn1 <- pr.nn1$net.result*(max(bank.df$PersonalLoan)-min(bank.df$PersonalLoan))+min(bank.df$PersonalLoan)
  
  test.cv.r1 <- (test.cv1$PersonalLoan)*(max(bank.df$PersonalLoan)-min(bank.df$PersonalLoan))+min(bank.df$PersonalLoan)
  
  cv.error1[i] <- sum((test.cv.r1 - pr.nn1)^2)/nrow(test.cv1)
  
  pbar1$step()
}

# Average MSE
mean(cv.error1) # 0.01627887694
median(cv.error1) # 0.01378789799
# MSE vector from CV
cv.error1

# Visual plot of CV results
boxplot(cv.error1,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)


par(mfrow=c(2,1))
hist(AA, col=rgb(0,0,1,1/4), xlim=c(0.013,0.04),ylim=c(0,1.5), main="ANN tanh (blue)\n ANN logistic (pinkish)\n ANN logistic and GLM overlap (orange)",xlab="Bank Data MSE with ANN analysis")
hist(BB, col=rgb(0.5,0,0.3,1/4), add=T)
hist(CC, col=rgb(1,0.5,1/4), add=T)
box()

boxplot(cv.error1,xlab='Bank Data MSE from Cross-Validation',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)

par(mfrow=c(1,1))

# Boxplot for both dataset together for Report:

par(mfrow=c(2,1))
# First dataset
boxplot(cv.error,xlab='School Data Cross-validation MSE',col='cyan',
        border='blue',horizontal=TRUE)

# Second dataset
boxplot(cv.error1,xlab='Bank Data Cross-validation MSE',col='cyan',
        border='blue',horizontal=TRUE)
par(mfrow=c(1,1))



# Barchart for both dataset for Report

par(mfrow=c(2,1))
hist(A, col=rgb(0.2,0.8,1/4),xlim=c(0.0,0.6), ylim=c(0.15,2.), main="NN tanh (green), NN logistic (pink) \n NN tanh and NN logistic overlap (dark green) \n Tanh, logistic, GLM overlap (coffee color)",xlab="School Data MSE with ANN analysis")
#hist(B, col=rgb(1,0,0,1/4), add=T)
hist(B, col=rgb(1,0,0.4,1/4), add=T)
hist(C, col=rgb(0.2,0,0,1/2), add=T)
box()

hist(AA, col=rgb(0,0,1,1/4), xlim=c(0.013,0.022),ylim=c(0,1.5), main="ANN tanh (blue)\n ANN logistic (brown)\n ANN logistic and GLM overlap (orange)",xlab="Bank Data MSE with ANN analysis")
hist(BB, col=rgb(0.5,0,0.3,1/4), add=T)
hist(CC, col=rgb(1,0.5,1/4), add=T)
box()
par(mfrow=c(1,1))

######################################################################
######################## End of ANN of second dataset bank.df ########
####################### Start of KNN of second dataset bank.df #######
######################################################################

set.seed(102)
# Explore data
dim(trainScaled1)
dim(testScaled1)
names(trainScaled1)
head(trainScaled1)
head(testScaled1)
trainScaled1$PersonalLoan

trainScaled1$PersonalLoan=ifelse(trainScaled1$PersonalLoan==0,"X0","X1") # change the levels to X0 and X1 from 0 and 1
testScaled1$PersonalLoan=ifelse(testScaled1$PersonalLoan==0,"X0","X1")
trainScaled1$PersonalLoan
testScaled1$PersonalLoan

# Setting levels for both training and validation data
levelsBank_trainScaled1<- make.names(levels(factor(trainScaled1$PersonalLoan)))
levelsBank_testScaled1 <- make.names(levels(factor(testScaled1$PersonalLoan)))
levelsBank_trainScaled1
levelsBank_testScaled1

# Setting up train controls
repeats = 3
numbers = 10
tunel = 20
numb=60
set.seed(123)
xBank = trainControl(method = "repeatedcv",
                       number = numbers,
                       repeats = repeats,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)

modelBank <- train(PersonalLoan~. , data = trainScaled1, method = "knn",
                     preProcess = c("center","scale"),
                     trControl = xBank,tuneGrid = expand.grid(k = 1:numb),
                     metric = "ROC", 
                     tuneLength = tunel)

# Summary of model
summary(modelBank)
modelBank
plot(modelBank,main='Bank Data')

# Testing
test_predBank <- predict(modelBank,testScaled1, type = "prob")
test_predBank

# For confusion matrix
test_predBank_1 <- predict(modelBank,testScaled1)
test_predBank_1
confusionMatrix(test_predBank_1, as.factor(testScaled1$PersonalLoan))
# 
# Confusion Matrix and Statistics
#              Reference
# Prediction   0   1
#         0 1342   96
#         1    5   57
# 
# Accuracy : 0.9327          
# 95% CI : (0.9188, 0.9448)


#Storing Model Performance Scores
testScaled1$PersonalLoan
testScaled1$PersonalLoan1=ifelse(testScaled1$PersonalLoan=="X0",0,1) # change the levels back to 0 and 1
pred_valBank <-prediction(test_predBank[,2],testScaled1$PersonalLoan1)

# Calculating Area under Curve (AUC)
perf_valBank <- performance(pred_valBank,"auc")
perf_valBank

# Plot AUC
perf_valBank <- performance(pred_valBank, "tpr", "fpr")
plot(perf_valBank, col = "blue", lwd = 1.5,main="Bank Data ROC", xlab='False positive rate (or 1-specificity)', ylab='True positive rate (or sensitivity)')

#Calculating KS statistics (KS for Kolmogorov-Smirnov)
ksBank <- max(attr(perf_valBank, "y.values")[[1]] - (attr(perf_valBank, "x.values")[[1]]))
ksBank  # 0.8113392348

# Calculate the Area under curve (AUC) on validation dataset
attributes(performance(pred_valBank, 'auc'))$y.values[[1]]  # Area = 0.9685394212


# Some plots together for Report:
plot(modelSchool, lwd=2, main='School Data') # default color is blue
plot(modelBank,col='blue', lwd=2, main='Bank Data')

# Some plots together for Report:
par(mfrow=c(2,1))
plot(perf_valSchool, col = "dark red", main="School Data ROC")
plot(perf_valBank, col = "blue", main="Bank Data ROC")
par(mfrow=c(1,1))

################### End ##################






