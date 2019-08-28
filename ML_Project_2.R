################################################################
# Ramesh Subedi

# Implement three learning algorithms:
# 1. Support Vector Machines (SVM)
# 2. Decision trees, and
# 3. Boosting
 
# Tasks:
# 1. You are to use two data sets for this work.
# a. First data set: Use the student performance data set from student-mat.csv as the 
# first data set. Use the converted case (binary classification).
# b. Second data set: Get a data set suitable for classification from anywhere 
# (either publicly available or your own). The data set should have a reasonable 
# amount of features and instances. You need to explain why you think this data set 
# and the corresponding classification problem is interesting.
# 
# Divide data sets in train and test sets.
# 
# 2. Download and use any support vector machines package to classify your classification problems. 
# Do it in such a way that you are able to easily change kernel functions. Experiment with at least 
# two kernel functions (in addition to linear) of your choice. You can pick any kernels you 
# like (shown in the class or not).
# 3. Download and use any decision trees package to classify your classification problems. 
# Experiment with pruning. You can use information gain or GINI index or any other metric to 
# split on variables. Just be clear to explain why you used the metric that you used.
# 4. Implement (or download) a package to use a boosted version of your decision trees. 
# Again, experiment with pruning.


###############################################################

rm(list=ls()) #drop all variables

######################################
library(data.table) 
library(magrittr)
library(dtplyr) 
library(sandwich) # for White correction
#library(lmtest) # for more advanced hypothesis testing tools
#library(tseries) # time series package
library(DBI) 
library(RSQLite) 
library(tidyverse)
library(broom)  # for tidy() function
#library(TSA)
#library(forecast)
library(vars)
#library(fpp) # for VAR forecast
#library(UsingR)
#library(margins)
#library(plm) # for pooled OLS (ordinary least squares)
library(car) # for scatterplot()
#library(aod) # for probit link
library(gradDescent) # for Gradient Descent calculation
library(glmnet)
library(e1071) # for Support Vector Machine, Titanic data, etc.
library(tree) # for tree to work on Decisiion Trees
library(gbm) # for gbm (gradient boosting model)
library(adabag) # for bagging
library(rpart) # 
library(party) # Recursive partitioning
library(partykit) # Pruning tree from party


problemData1 <- read.table("~/mlData/data2/student-mat.csv",sep=";",header=TRUE)
names(problemData1)


################ TASK 1a #######################

# split whole problemData1 data into 70% for training and 30% for testing.

set.seed(1) # set fixed seed so that radom sampling for splitting data in 70/30 ratio is reproducible.

# Randomly sample 70% of the row IDs for training
train.rows <- sample(rownames(problemData1), dim(problemData1)[1]*0.7)

# assign the remaining 30% row IDs serve as test
test.rows <- sample(setdiff(rownames(problemData1), train.rows),dim(problemData1)[1]*0.3)

# create the 3 data frames by collecting all columns from the appropriate rows
train.data <- problemData1[train.rows, ]
test.data <-  problemData1[test.rows, ]

class(train.data)
names(train.data)
train.data$school

train.data <- train.data %>% dplyr::select(-G1,-G2) # Drop G2 and G2 variables from train.data

# Convert Factors into Numeric

names(train.data)
train.data$school
str(train.data$school)
train.data$school <- as.numeric(train.data$school)
train.data$school
str(train.data$school)

train.data$sex <- as.numeric(train.data$sex)
train.data$address<-as.numeric(train.data$address)
train.data$famsize <- as.numeric(train.data$famsize)
train.data$Pstatus<- as.numeric(train.data$Pstatus)
train.data$Fjob<-as.numeric(train.data$Fjob)
train.data$Mjob<- as.numeric(train.data$Mjob)
train.data$reason<- as.numeric(train.data$reason)
train.data$guardian<- as.numeric(train.data$guardian)
train.data$schoolsup<- as.numeric(train.data$schoolsup)
train.data$famsup<- as.numeric(train.data$famsup)
train.data$paid<- as.numeric(train.data$paid)
train.data$activities<- as.numeric(train.data$activities)
train.data$nursery<- as.numeric(train.data$nursery)
train.data$higher<- as.numeric(train.data$higher)
train.data$internet<- as.numeric(train.data$internet)
train.data$romantic<- as.numeric(train.data$romantic)


train.data%>%mutate_if(is.numeric, scale) # This scaling works. Though scaling is only for numeric variables (not for factors), we changed factors into numeric above. Hence this scaling works for all variables.

names(train.data)
class(train.data) # To check if train.data is still data.frame

# Repeat the same thing with test data.
test.data <- test.data %>% dplyr::select(-G1,-G2) # Drop G2 and G2 variables from test.data

names(test.data)
test.data$school
str(test.data$school)
test.data$school <- as.numeric(test.data$school)
test.data$school
str(test.data$school)
test.data$sex <- as.numeric(test.data$sex)
test.data$address<-as.numeric(test.data$address)
test.data$famsize <- as.numeric(test.data$famsize)
test.data$Pstatus<- as.numeric(test.data$Pstatus)
test.data$Fjob<-as.numeric(test.data$Fjob)
test.data$Mjob<- as.numeric(test.data$Mjob)
test.data$reason<- as.numeric(test.data$reason)
test.data$guardian<- as.numeric(test.data$guardian)
test.data$schoolsup<- as.numeric(test.data$schoolsup)
test.data$famsup<- as.numeric(test.data$famsup)
test.data$paid<- as.numeric(test.data$paid)
test.data$activities<- as.numeric(test.data$activities)
test.data$nursery<- as.numeric(test.data$nursery)
test.data$higher<- as.numeric(test.data$higher)
test.data$internet<- as.numeric(test.data$internet)
test.data$romantic<- as.numeric(test.data$romantic)

test.data%>%mutate_if(is.numeric, scale) 

# For logistic regression, we make G3 as a boolean variable with this criteria:
# If G3>meanValueOfG3, take G3 as 1, otherwise take G3 as 0.
# This is how we do it (use ifelse(test,yes,no) statement which is ifelse(test,1,0)).
grade <- train.data$G3
grade
meanVal<- mean(train.data$G3)
meanVal
myGrade <- ifelse(grade>=meanVal,1,0)
myGrade
grade
plot(myGrade,xlab='Student Number',ylab='Final Grade (G3)')

xdata <- train.data %>% dplyr::select(-G3) # Drop G3 since it's new name is grade.

names(xdata)
summary(xdata)
xdata%>%mutate_if(is.numeric, scale) # scales the numeric data, leaves non-numeric alone.
# For the test.data:

grade1 <-test.data$G3
grade1
meanVal1<-mean(test.data$G3)
meanVal1
myGrade1 <- ifelse(grade1>=meanVal1,1,0)
myGrade1
grade1
plot(myGrade1,xlab='Student Number',ylab='Final Grade (G3)')

xdata1 <- test.data %>% dplyr::select(-G3) # Drop G3 since it's new name is grade.

summary(xdata1)
xdata1%>%mutate_if(is.numeric, scale) # scales the numeric data, leaves non-numeric alone.


#######################################
# kernel = 'linear'
# kernel = 'radial'
# kernel = 'polynomial'
# kernel = 'sigmoid'  (NOT USED in the analysis)

names(xdata)
dim(xdata)
length(myGrade)

myData = data.frame(x=xdata,y=as.factor(myGrade))

names(myData)
myData$y
length(myData$y) # 276

# find optimal cost of misclassification
tune.out <- tune(svm, y~., data = myData, kernel = "linear",ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# extract the best model
bestmod <- tune.out$best.model
bestmod # cost = 0.1 with type='C-classification' 
out <- svm(myGrade~., data=myData,kernel="linear", cost=0.1,type='C-classification',scale=TRUE) # C-classification is a must, but scale is immaterial in our case as the data were already scaled.

# It would be impossible to visualize such data since there are more than 2 features.
# Hence no plots are possible like this type: plot(out,myData)
out$index
summary(out)
# Call:
#   svm(formula = myGrade ~ ., data = myData, kernel = "linear", cost = 0.1, type = "C-classification", scale = TRUE)
# Parameters:
# SVM-Type:  C-classification 
# SVM-Kernel:  linear 
# cost:  0.1 
# gamma:  0.03125 
# Number of Support Vectors:  88
# ( 49 39 )
# Number of Classes:  2 
# Levels: 
#   0 1

#table(out$fitted,myData$y)
confusionMatrix(out$fitted,myData$y)
# Here is the Confusion Matrix
#             Reference
# Prediction  0     1
#          0  126   0
#          1  0     150
#
# This shows no training error (0/276 = 0% error). All 276 observations are 
# correctly classified which is not unusual for a training data 
# because of large number of variables (30)
# compared to the number of observations (276)).

# Let's repeat the same for radial kernel.

# find optimal cost of misclassification
tune.out1 <- tune(svm, y~., data = myData, kernel = "radial",ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# extract the best model
bestmod1 <- tune.out1$best.model
bestmod1 # type='C-classification',cost=1,gamma=0.03333333

out1 <- svm(myGrade~., data=myData,kernel="radial",type='C-classification',cost=1,gamma=0.03333333)
out1$index
summary(out1)
# Call:
#   svm(formula = myGrade ~ ., data = myData, kernel = "radial", type = "C-classification", 
#       cost = 1, gamma = 0.03333333)
# Parameters:
# SVM-Type:  C-classification 
# SVM-Kernel:  radial 
# cost:  1 
# gamma:  0.03333333 
# Number of Support Vectors:  238
# ( 120 118 )
# Number of Classes:  2 
# Levels: 
#   0 1
#table(out1$fitted,myData$y)
confusionMatrix(out1$fitted,myData$y)
# Here is the same Confusion Matrix again as in for linear kernel:
#              Reference
# Prediction   0     1
#           0  126   0
#           1  0     150
#

# Now let's try for a polynomial kernel

# find optimal cost of misclassification
tune.out2 <- tune(svm, y~., data = myData, kernel = "polynomial",ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# extract the best model
bestmod2 <- tune.out2$best.model
bestmod2 # type='C-classification',cost=1,degree=3,gamma=0.03333333

out2 <- svm(myGrade~., data=myData,kernel="polynomial",type='C-classification',cost=5,degree=3,gamma=0.03333333)
out2$index
summary(out2)
# Call:
#   svm(formula = myGrade ~ ., data = myData, kernel = "polynomial", type = "C-classification", 
#       cost = 1, degree = 3, gamma = 0.03333333)
# Parameters:
# SVM-Type:  C-classification 
# SVM-Kernel:  polynomial 
# cost:  1 
# degree:  3 
# gamma:  0.03333333 
# coef.0:  0 
# Number of Support Vectors:  262
# ( 145 117 )
# Number of Classes:  2 
# Levels: 
#   0 1

#table(out2$fitted,myData$y)
confusionMatrix(out2$fitted,myData$y)
# The Confusion Matrix is different now, 14 data are misclassified as true given 
# but predicted true:
# Vertically   (0 for false given (reference), 1 for true given (reference)).
# Horizontally (0 for false prediction, 1 for true prediction).
#             Reference
# Prediction  0     1
#          0  126   0
#          1  0    150
#

# Out of 276 data points for the training data set, 
# the linear kernel identifies 88 support vectors (49 39),
# the radial kernel identifies 238 support vectors (120 118),
# the polynomial kernel identifies 245 support vectors (135 110).
# Hence the polynomial kernel appears to be the best identifier
# among the three.

# Let's repeat the same thing for the test data set.

myData1 = data.frame(x=xdata1,y=as.factor(myGrade1))
names(myData1)
myData1$y
length(myData1$y) # 118

outTest <- svm(myGrade1~., data=myData1,kernel="linear", cost=0.1,type='C-classification',scale=TRUE)
outTest$index
summary(outTest)
# Call:
#   svm(formula = myGrade1 ~ ., data = myData1, kernel = "linear", cost = 0.1, type = "C-classification", scale = TRUE)
# Parameters:
# SVM-Type:  C-classification 
# SVM-Kernel:  linear 
# cost:  0.1 
# gamma:  0.03125 
# Number of Support Vectors:  56
# ( 22 34 )
# Number of Classes:  2 
# Levels: 
#   0 1

#table(outTest$fitted,myData1$y)
confusionMatrix(outTest$fitted,myData1$y)
# Confusion matrix
#           Reference
#Prediction 0  1
#         0 42 0
#         1 0  76
# All 118 test data observations are correctly classified.

# Now for radial kernel for test data
length(myGrade1) # 118
outTest1 <- svm(myGrade1~., data=myData1,kernel="radial",type='C-classification',cost=1,gamma=0.03333333)
outTest1$index
summary(outTest1)
# Call:
#   svm(formula = myGrade1 ~ ., data = myData1, kernel = "radial", type = "C-classification", 
#       cost = 1, gamma = 0.03333333)
# Parameters:
# SVM-Type:  C-classification 
# SVM-Kernel:  radial 
# cost:  1 
# gamma:  0.03333333 
# Number of Support Vectors:  106
# ( 41 65 )
# Number of Classes:  2 
# Levels: 
#   0 1

#table(outTest1$fitted,myData1$y)
confusionMatrix(outTest1$fitted,myData1$y)
#           Reference
#Prediction 0  1
#         0 42 0
#         1 0  76
# All 118 test data observations are correctly classified.
# Classification error is 0/118 =0 = 0%

# Now for polynomial kernel
outTest2 <- svm(myGrade1~., data=myData1,kernel="polynomial",type='C-classification',cost=5,degree=3,gamma=0.03333333)
outTest2$index
summary(outTest2)
#table(outTest2$fitted,myData1$y)
confusionMatrix(outTest2$fitted,myData1$y)
#           Reference
#Prediction 0  1
#         0 27 15
#         1 0  76
# Only 103 observations out of 118 test data observations are correctly classified.
# 15 false observations are predicted as true observations
# Hence classification error is 15/118 = 0.1271186 = 12.7%

# Only linear and radial kernels classify the 118 test data observations with zero error
# while the polynomial kernel classification gives 12.7% error.

# End of Support Vector Machine for first set of data (from studentMat.csv).

######################## Decision Tree #####################################

# Now we are using the same data studentMat.csv for Decision trees.

# Training data
newData <- data.frame(xdata,myGrade)
names(newData)
newData$failures
newData$schoolsup
newData$age
newData$studytime
#tree.newData<-tree(myGrade~.-myGrade, newData,method = "recursive.partition") # It seems that the default method is "recursive.partition."
tree.newData<-tree(myGrade~.-myGrade, newData)
plot(tree.newData)
text(tree.newData,pretty=0)

summary(tree.newData)
# Regression tree:
#   tree(formula = myGrade ~ . - myGrade, data = newData)
# Variables actually used in tree construction:
# "failures"   "schoolsup"  "age"        "Fjob"       "goout"      "absences"   "sex"       
# "studytime"  "reason"     "traveltime" "Mjob"       "freetime"   "paid"       "Fedu"      
# "health"    
# Number of terminal nodes:  24 
# Residual mean deviance:  0.1441 = 36.32 / 252 
# Distribution of residuals:
#   Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# -0.94440 -0.15000  0.05556  0.00000  0.11760  0.95000 

cvTrain <- cv.tree(tree.newData)  # check whether pruning improves
cvTrain
plot(cvTrain$size,cvTrain$dev,type='b')
plot(cvTrain$k,cvTrain$dev,type='b')
pruneTrain <- prune.tree(tree.newData,best=5)# Pruning
# Plot for REPORT
par(mfrow=c(1,2))
plot(tree.newData)
text(tree.newData,pretty=0)
plot(pruneTrain)
text(pruneTrain,pretty=0)
par(mfrow=c(1,1))

# This will calculate MSE
 m1 <- rpart(myGrade~.,data=xdata, method  = "anova",control = list(minsplit = 5, maxdepth = 12, xval = 10))
 print(m1)
 plot(m1)
 text(m1,pretty=0)
m1$cptable
pred<-predict(m1,newdata=xdata)
RMSE(pred=pred,myGrade) # 0.3886487
mse = (RMSE(pred=pred,myGrade)^2)
mse # 0.1510478
plotcp(m1)


# Now do the same thing for test data:

newDataTest <- data.frame(xdata1,myGrade1)
names(newDataTest)
tree.newDataTest<-tree(myGrade1~.-myGrade1, newDataTest)
plot(tree.newDataTest)
text(tree.newDataTest,pretty=0)
summary(tree.newDataTest)
# Regression tree:
# tree(formula = myGrade1 ~ . - myGrade1, data = newDataTest)
# Variables actually used in tree construction:
# "failures" "Fedu"     "Walc"     "famrel"   "absences" "Fjob"     "sex"      "goout"   
# "age"      "freetime" "health"  
# Number of terminal nodes:  14 
# Residual mean deviance:  0.12 = 12.48 / 104 
# Distribution of residuals:
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -0.8889 -0.0625  0.0000  0.0000  0.1111  0.9375 

cvTest <- cv.tree(tree.newDataTest)  # check whether pruning improves
cvTest
plot(cvTest$size,cvTest$dev,type='b')
plot(cvTest$k,cvTest$dev,type='b')
pruneTest <- prune.tree(tree.newDataTest,best=5)# Pruning
# Plot for REPORT
par(mfrow=c(1,2)) 
plot(tree.newDataTest)
text(tree.newDataTest,pretty=0)
plot(pruneTest)
text(pruneTest,pretty=0)
par(mfrow=c(1,1))
# To find MSE:
m2 <- rpart(myGrade1~.,data=xdata1, method  = "anova",control = list(minsplit = 5, maxdepth = 12, xval = 20))
print(m2)
plot(m2)
text(m2,pretty=0)
m2$cptable
pred<-predict(m2,newdata=xdata1)
RMSE(pred=pred,myGrade1) # 0.3737783
mse = (RMSE(pred=pred,myGrade1)^2)
mse #0.1397102
plotcp(m2)
# # Plot both plots in one frame
 par(mfrow=c(1,2))
 plot(tree.newData)
 text(tree.newData,pretty=0)
 plot(tree.newDataTest)
 text(tree.newDataTest,pretty=0)
 par(mfrow=c(1,1))

########################### Boosting ##################################

# Now we are using the same data studentMat.csv for Boosting.
# For training data
boostTrain <- gbm(myGrade~., data = xdata, distribution = "gaussian", n.trees = 5000, shrinkage = 0.015, interaction.depth = 5, cv.folds=5, n.cores = NULL, verbose=FALSE) # gradient boosting model (gbm)
print(boostTrain) 

# Find index for n trees with minimum CV error
min_MSE <- which.min(boostTrain$cv.error)
min_MSE # 89 trees with minimum Cross Validation error
# Get MSE
mse<-min(boostTrain$cv.error) # 0.2380618
mse
# Compute RMSE
sqrt(min(boostTrain$cv.error)) # 0.4879158


# Plot loss function as a result of n trees added to the ensemble
gbm.perf(boostTrain, method = "cv") # same plot as for print(boostTrain) or boostTrain

# Plots for REPORT
par(mfrow=c(1,2))
boostTrain
summary(boostTrain) 
par(mfrow=c(1,1))
plot(boostTrain,i="absences")
plot(boostTrain,i="Mjob")
cor(xdata$absences,myGrade) #  -0.08134072
cor(xdata$Mjob,myGrade) # 0.1146003


# For test data:
boostTest <- gbm(myGrade1~., data = xdata1, distribution = "gaussian", n.trees = 5000, shrinkage = 0.015, interaction.depth = 5, cv.folds=5, n.cores = NULL, verbose=FALSE) 
boostTest
#summary(boostTest) #Summary gives a table of Variable Importance and a plot of Variable Importance
# Plots for REPORT
par(mfrow=c(1,2))
boostTest
summary(boostTest) 
par(mfrow=c(1,1))
min_MSE <- which.min(boostTest$cv.error)
min_MSE # 51 trees with minimum Cross Validation error
# Get MSE
mse<-min(boostTest$cv.error) 
mse # 0.2181769
# Compute RMSE
sqrt(min(boostTest$cv.error)) # 0.4670941
plot(boostTest,i="absences")
plot(boostTest,i="failures")
cor(xdata1$absences,myGrade1) #  -0.009757308
cor(xdata1$failures,myGrade1) # -0.3861792




################ End of first data set ###############################
####### That is, end of work with data student-mat.csv ################
######################################################################





### Now we work everything as above with the second data set #########
######################################################################
# Dataset from here:
# https://github.com/gchoi/Dataset/blob/master/UniversalBank.csv

Bank.df <- read.csv("~/mlData/data2/bank.csv", header = TRUE) 
names(Bank.df)
str(Bank.df)
Bank.df$PersonalLoan # already a binary variable
Bank.df$CDAccount
dim(Bank.df) # 5000   14

# Imputing data (no need here)
#bank.df <- na.omit(bank.df) # ommit all NA's
#sum(is.na(bank.df$PersonalLoan)) # 0 means no missing data

bank.df <- Bank.df[ , -c(1, 5)]  # Drop ID and zip code columns.
names(bank.df)
dim(bank.df) # 5000   12

# transform PersonalLoan into categorical variable
bank.df$PersonalLoan = as.factor(bank.df$PersonalLoan)
# partition the data where 70% is train.df and rest is test.df.
train.index <- sample(c(1:dim(bank.df)[1]), dim(bank.df)[1]*0.7)
train.df <- bank.df[train.index, ]
test.df <- bank.df[-train.index, ]

train.df$PersonalLoan
table(train.df$PersonalLoan)
plot(train.df$PersonalLoan)



# Support Vector Machine for bank data ##############

# For training data
# find optimal cost of misclassification for linear kernel
tuneBank <- tune(svm, PersonalLoan~., data = train.df, kernel = "linear",ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# extract the best model
bestmod <- tuneBank$best.model
bestmod 
outBank <- svm(PersonalLoan~., data=train.df,kernel="linear", cost=1,type='C-classification',scale=TRUE) 

# It would be impossible to visualize such data since there are more than 2 features. Hence no plots are possible like this type: plot(out,myData)
outBank$index
summary(outBank)
#table(train.df$PersonalLoan,outBank$fitted)
confusionMatrix(train.df$PersonalLoan,outBank$fitted)
# Here is the Confusion Matrix
#             Reference
# Prediction  0      1
#          0  3119   25
#          1  131    225
# Accuracy : 0.9546    (error = 1 - Accuracy = 1-0.9546 =0.0454=4.54%)      
# 95% CI : (0.9471, 0.9612)

# find optimal cost of misclassification for radial kernel
tuneBank1 <- tune(svm, PersonalLoan~., data = train.df, kernel = "radial",ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# extract the best model
bestmod1 <- tuneBank1$best.model
bestmod1 
outBank1 <- svm(PersonalLoan~., data=train.df,kernel="radial", cost=5,type='C-classification',gamma=0.09090909,scale=TRUE) 
outBank1$index
summary(outBank1)
confusionMatrix(train.df$PersonalLoan,outBank1$fitted)
# Confusion Matrix and Statistics
#               Reference
# Prediction    0      1
#          0    3138   6
#          1    36     320
# Accuracy : 0.988, (error = 1-Accuracy=1-0.988=0.012=1.2%)          
# 95% CI : (0.9838, 0.9913)

# find optimal cost of misclassification for polynomial kernel
tuneBank2 <- tune(svm, PersonalLoan~., data = train.df, kernel = "polynomial",ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# extract the best model
bestmod2 <- tuneBank2$best.model
bestmod2 
outBank2 <- svm(PersonalLoan~., data=train.df,kernel="polynomial", cost=10,type='C-classification',gamma=0.09090909,degree=3,scale=TRUE) 
outBank2$index
summary(outBank2)
confusionMatrix(train.df$PersonalLoan,outBank2$fitted)
# Confusion Matrix and Statistics
#               Reference
# Prediction    0     1
#          0    3141  3
#          1    36    320
# Accuracy : 0.9889  (error = 1 - Accuracy = 1-0.9889=0.0111=1.11%)        
# 95% CI : (0.9848, 0.9921)



# Do the same thing for test data

outBankTest <- svm(PersonalLoan~., data=test.df,kernel="linear", cost=1,type='C-classification',scale=TRUE) 
outBankTest$index
summary(outBankTest)
confusionMatrix(test.df$PersonalLoan,outBankTest$fitted)
# Confusion Matrix and Statistics
#               Reference
# Prediction    0      1
#          0    1369   7
#          1    56     68
# Accuracy : 0.9513          
# 95% CI : (0.9392, 0.9617)

outBankTest1 <- svm(PersonalLoan~., data=test.df,kernel="radial", cost=5,type='C-classification',gamma=0.09090909,scale=TRUE) 
outBankTest1$index
summary(outBankTest1)
confusionMatrix(test.df$PersonalLoan,outBankTest1$fitted)
# Confusion Matrix and Statistics
#              Reference
# Prediction   0     1
#          0   1375  1
#          1   12    112
# Accuracy : 0.9913, (error = 1-Accuracy=1-0.9913=0.0087=0.87%)        
# 95% CI : (0.9852, 0.9954)


outBankTest2 <- svm(PersonalLoan~., data=test.df,kernel="polynomial", cost=10,type='C-classification',gamma=0.09090909,degree=3,scale=TRUE) 
outBankTest2$index
summary(outBankTest2)
confusionMatrix(test.df$PersonalLoan,outBankTest2$fitted)
# Confusion Matrix and Statistics
#              Reference
# Prediction   0       1
#          0   1376    0
#          1   14      110
# Accuracy : 0.9907, (error = 1-Accuracy=1-0.9907=0.0093=0.93%)     
# 95% CI : (0.9844, 0.9949)




# Decesion tree for bank data

treeBankTrain <- tree(PersonalLoan ~ ., data = train.df)
names(train.df)
predTrain <- predict(treeBankTrain, test.df, type = "class")
confusionMatrix(predTrain, test.df$PersonalLoan)
summary(treeBankTrain)
plot(treeBankTrain)
text(treeBankTrain,pretty=0)
cvBankTrain <- cv.tree(treeBankTrain)  # check whether pruning improves
cvBankTrain
plot(cvBankTrain$size,cvBankTrain$dev,type='b')
plot(cvBankTrain$k,cvBankTrain$dev,type='b')
pruneBankTrain <- prune.tree(treeBankTrain,best=5)# Pruning
plot(pruneBankTrain)
text(pruneBankTrain,pretty=0)
# decision tree plot for REPORT for training data:
par(mfrow=c(1,2))
plot(treeBankTrain)
text(treeBankTrain,pretty=0)
plot(pruneBankTrain)
text(pruneBankTrain,pretty=0)
par(mfrow=c(1,1))



# Now do the same thing for test data:

treeBankTest <- rpart(PersonalLoan ~ ., data = test.df)
predBankTest <- predict(treeBankTest, train.df, type = "class")
confusionMatrix(predBankTest, train.df$PersonalLoan)
summary(treeBankTest) #Summary gives a table of Variable Importance and a plot of Variable Importance

treeBankTest <- tree(PersonalLoan ~ ., data = test.df)
predBankTest <- predict(treeBankTest, train.df, type = "class")
confusionMatrix(predBankTest, train.df$PersonalLoan)
summary(treeBankTest)
cvBankTest <- cv.tree(treeBankTest)  # check whether pruning improves
cvBankTest
plot(cvBankTest$size,cvBankTest$dev,type='b')
plot(cvBankTest$k,cvBankTest$dev,type='b')
pruneBankTest <- prune.tree(treeBankTest,best=5)# Pruning

par(mfrow=c(1,2))
plot(treeBankTest)
text(treeBankTest,pretty=0)
plot(pruneBankTest)
text(pruneBankTest,pretty=0)
par(mfrow=c(1,1))


# Boosting for bank data: trining data

boostBankTrain <- gbm(PersonalLoan~ . ,data = train.df,distribution = "gaussian",n.trees = 10000,shrinkage = 0.01, interaction.depth = 4, cv.folds=5, n.cores = NULL, verbose=FALSE)
boostBankTrain
summary(boostBankTrain) 
plot(boostBankTrain,i="Income")
plot(boostBankTrain,i="Education")
cor(train.df$Income,as.numeric(train.df$PersonalLoan)) #  0.5134711
cor(train.df$Education,as.numeric(train.df$PersonalLoan)) # 0.1475339

# Plots for REPORT
par(mfrow=c(1,2))
boostBankTrain
summary(boostBankTrain) 
par(mfrow=c(1,1))


# MSE
min_MSE_BankTrain <- which.min(boostBankTrain$cv.error)
min_MSE_BankTrain # 
# Get MSE
mseBankTrain<-min(boostBankTrain$cv.error) # 
mseBankTrain # MSE = 0.01096052
# Compute RMSE
sqrt(min(boostBankTrain$cv.error)) # RMSE = 0.1046925



# Boosting for bank data: test data
boostBankTest <- gbm(PersonalLoan~ . ,data = test.df,distribution = "gaussian",n.trees = 10000,shrinkage = 0.01, interaction.depth = 4, cv.folds=5, n.cores = NULL, verbose=FALSE)
boostBankTest
summary(boostBankTest)

# Plots for REPORT
par(mfrow=c(1,2))
boostBankTest
summary(boostBankTest) 
par(mfrow=c(1,1))

# MSE
min_MSE_BankTest <- which.min(boostBankTest$cv.error)
min_MSE_BankTest # 
# Get MSE
mseBankTest<-min(boostBankTest$cv.error) # 
mseBankTest # MSE = 0.01282524
# Compute RMSE
sqrt(min(boostBankTest$cv.error)) # RMSE = 0.1132486

################### End ##############




