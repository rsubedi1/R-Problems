################################################################
# Ramesh Subedi

# Implement the following clustering algorithms: 
# 1. K-means
# 2. Expectation Maximization
# 
# In addition, implement the following feature dimensionality reduction algorithms
# 1. Any one feature selection algorithm (decision tree, forward selection, backward elimination,etc.)
# 2. PCA
# 3. ICA
# 4. Randomized Projections
# 
# 
# Tasks:
# 1. Run the clustering algorithms on your datasets and describe your observations (with plots).
# 2. Apply the dimensionality reduction algorithms on your datasets and describe your observations
# (with plots).
# 3. Run the clustering algorithms again, this time after applying dimensionality reduction. Describe
# the difference compared to previous experimentation (with plots).
# 4. Run your neural network learner from assignment 3 on the data after dimensionality reduction
# (from task 2). Explain and plot your observations (error rates, etc.)
# 5. Use the clustering results from task 1 as the new features and apply neural network learner on
# this new data consisting of only clustering results as features and class label as the output. 
# Again, plot and explain your results.



###############################################################

######################################
library(data.table) 
library(magrittr)
library(plyr) 
library(dtplyr) 
library(sandwich) # for White correction
library(lmtest) # for more advanced hypothesis testing tools
#library(tseries) # time series package
#library(DBI) 
#library(RSQLite) 
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
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
library(glmnet)
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
library(scales)
library(factoextra)
library(FactoInvestigate)
library(FactoMineR) 
library(flexclust)  # to quantify performance of kmeans 
library(NbClust)
library(mclust) # for Expectation Maximization Clustering
library(ica) # for ICA (independent component analysis)
library(fastICA) # for fast ICA (independent component analysis)
library(vegan) # for hclust (hierarchical clustering) of some type
library(ggbiplot)
library(fpc) # for plotcluster
library(RandPro) # for randomized projection

# rm(list=ls()) #drop all variables
# start with a clean slate
rm(list=ls(all=TRUE)) 

data <- read.table("~/mlData/student-mat.csv",sep=";",header=TRUE)

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

Data <-data  # Data is unscaled
names(Data)
Data$G3

# Normalize data (scaling)

normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
names(data)
data$school <- normalize(data$school)
data$sex <- normalize(data$sex)
data$age <- normalize(data$age)
data$address <- normalize(data$address)
data$famsize <- normalize(data$famsize)
data$Pstatus <- normalize(data$Pstatus)
data$Medu <- normalize(data$Medu)
data$Fedu <- normalize(data$Fedu)
data$Mjob <- normalize(data$Mjob)
data$Fjob <- normalize(data$Fjob)
data$reason <- normalize(data$reason)
data$guardian <- normalize(data$guardian)
data$traveltime <- normalize(data$traveltime)
data$studytime <- normalize(data$studytime)
data$failures <- normalize(data$failures)
data$schoolsup <- normalize(data$schoolsup)
data$famsup <- normalize(data$famsup)
data$paid <- normalize(data$paid)
data$activities <- normalize(data$activities)
data$nursery <- normalize(data$nursery)
data$higher <- normalize(data$higher)
data$internet <- normalize(data$internet)
data$romantic <- normalize(data$romantic)
data$famrel <- normalize(data$famrel)
data$freetime <- normalize(data$freetime)
data$goout <- normalize(data$goout)
data$Dalc <- normalize(data$Dalc)
data$Walc <- normalize(data$Walc)
data$health <- normalize(data$health)
data$absences <- normalize(data$absences)
data$G3 <- normalize(data$G3)

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


############# TASK1 ########################################
############ K-Means from here #############################

set.seed(20)
Clusters <- kmeans(data, centers=10, nstart=50)
Clusters

# So how well did the K-means clustering uncover the actual structure of the data contained in the MyGrade variable? A cross-tabulation of MyGrade and cluster membership is given by
confMat <- table(data$MyGrade, Clusters$cluster)
confMat
# We can quantify the agreement between MyGrade and cluster, using an adjusted Rank index provided by the flexclust package.
randIndex(confMat) # 0.1038752 for 10 clusters
# The adjusted Rand index provides a measure of the agreement between two partitions, adjusted for chance. It ranges from -1 (no agreement) to 1 (perfect agreement). 
set.seed(20)
ClustersFinal <- kmeans(data, 8, nstart=25)
confMatFinal <- table(data$MyGrade, ClustersFinal$cluster)
randIndex(confMatFinal) # 0.1685834  for 8 clusters (optimal cluster # from scree plot)

dataKmeans <- as.data.frame(ClustersFinal$centers) # This is the output data from kmeans with 8 clusters

#### standalone code to plot scree plot for kmeans ###############
wssplot <- function(data, nc=20, seed=1234){  # nc for number of centers
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i,nstart=1)$withinss)}
  # For nstart>4 we did not see elbow, hence accepted R's default nstart=1 to observe elbow.
  plot(1:nc, wss, type="b", xlab="Number of Clusters", ylab="Variance explained")}
wssplot(data) # best elbow plot, cluster number=8
abline(v = 8, lty =2) # draw a veritcal dashed line from 8th cluster
############# kmeans scree plot done ###############################


#### Expectation Maximization ##################
dataDefault <- mclustBIC(data)
dataCustomize <- mclustBIC(data, G = 1:20, x = dataDefault)
plot(dataCustomize, G = 3:20, legendArgs = list(x = "topright"),xlab='Number of clusters') # best value at 7th cluster
abline(v = 7, lty =2) 


############### TASK2 ##############################################
##### 1. For Decision Tree  use hclust (hierarchical clustering) ###

hcluster <- hclust(dist(data), method='complete')
hclustTrim <- cutree(hcluster,8)
plot(hclustTrim)

Eucl_EX<-vegdist(data,"euclidean") 
Eucl<-hclust(Eucl_EX,method="complete") # The mothod is 'complete' linkage
# plot for report
plot(Eucl,main='Complete Linkage Dendrogram',xlab='')

chord_EX<-vegdist(decostand(data,"norm"),"euclidean") 
chord_EX
chord<-hclust(chord_EX,method="complete") 
plot(chord)

# Cut tree so that it has 8 clusters
hclust.clusters <- cutree(chord, k = 8)
plot(hclust.clusters)


# Compare cluster membership to actual MyGrade
table(hclust.clusters, MyGrade)
sum(apply(table(hclust.clusters, MyGrade), 1, min)) # 146

# To compare with kmeans
abc<-kmeans(scale(data), centers = 8)
table(abc$cluster,MyGrade)

# Table for Report:
table(hclust.clusters, MyGrade)  # from hclust
table(abc$cluster,MyGrade)  # from kmeans


#################### 2. PCA using prcomp, and PCA for Investigate() utility ###########

pca <- prcomp(data,scale=T) 
plot(pca)
plot(pca$sdev^2,xlab="Principal Components",ylab="Variance Explained")
plot(cumsum(pca$sdev^2),xlab="Principal Components",ylab="Cumulative Variance Explained")

# A few plots for report from this:
PCA1 <- PCA(data,ncp=10,graph=T, scale.=T) 
Investigate(PCA1) 

######## output data from PCA ##############
dataPCA <- as.data.frame(pca$x) # This is the output data from prcomp analysis
head(dataPCA[,1:2])
plot(PC1~PC2, data=dataPCA, cex = 0.5, lty = "solid")
text(PC1~PC2, data=dataPCA, labels=rownames(data),cex=.8)
##############################################

biplot(pca,scale=0) # stick with scale=0
pcavar<- pca$sdev^2
pve<-pcavar/sum(pcavar)
ggbiplot(pca,circle=T) # shows only dots

colMeans(data) # Mean of each variable
apply(data,2,sd) # Standard deviation of each variable

# Plot for report
par(mfrow=c(2,1))
plot(pve,xlab='Principal Components', ylab='Proportion of variance explained',ylim=c(0.01,0.12)) # this is like elbow plot (elbow at 7)
abline(v = 7, lty =2)
plot(cumsum(pve),xlab='Principal Components', ylab='Cumulative Proportion of variance explained',ylim=c(0.01,1)) # this is like elbow plot (elbow at 7)
abline(v = 7, lty =2)
abline(h = 0.43, lty =2)
par(mfrow=c(1,1))

# The total of 42.961% variance explained by 7 principal components


############### ICA  ############

icaModel <- fastICA(data, 8, alg.typ = "deflation", fun = "logcosh", alpha = 1,
                    method = "R", row.norm = FALSE, maxit = 200,
                    tol = 0.0001, verbose = F)

# plot for report
par(mfrow = c(1, 3))
plot(icaModel$X, main = "Original Component",xlim=c(-2,3),ylim=c(-2.5,2.5))
plot(icaModel$X %*% icaModel$K, main = "PCA components",xlim=c(-2,3),ylim=c(-2.5,2.5))
plot(icaModel$S, main = "ICA components",xlim=c(-2,3),ylim=c(-2.5,2.5))
par(mfrow = c(1, 1))

names(icaModel)
summary(icaModel)
icaModel$X # pre-processed data matrix 
icaModel$K # pre-whitening matrix that projects data onto the first n.comp principal components.
icaModel$W # estimated un-mixing matrix  (no.factors by no.signals)  - no. means number of 
icaModel$A # estimated mixing matrix (no.signals by no.factors) - no. means number of 
icaModel$S # estimated source matrix (The column vectors of estimated independent components (no.obs by no.factors)) - here no.factors means number of principal components

# check whitening:
# check correlations are zero
cor(icaModel$K) # correlations are not zero
# check diagonals are 1 in covariance
cov(icaModel$K) # diagonals are not 1

cor(icaModel$X) 
cor(icaModel$W) 
#cor(icaModel$A)

# table for report
cor(icaModel$S)  # no correlations
cov(icaModel$S)  # diagonal elements are 1


############## Randomized Projections ##################
# https://cran.r-project.org/web/packages/RandPro/RandPro.pdf
set.seed(101)
index.data <- sample(1:nrow(data),round(0.70*nrow(data)))
Train <- data[index.data,] # 70%
Test <- data[-index.data,] # 30%
trainl<-as.factor(Train$MyGrade) # need to declare as factor.
testl<-as.factor(Test$MyGrade)  # need to declare as factor.
train<-Train[,1:30]
test<-Test[,1:30]
randomProjection<-classify(train, test, trainl, testl)

# Confusion Matrix and Statistics
#             Reference
# Prediction  0   1
#       0     26  12
#       1     28  53
# 
# Accuracy : 0.6639          
# 95% CI : (0.5715, 0.7478)
# No Information Rate : 0.5462          
# P-Value [Acc > NIR] : 0.006031        
# 
# Kappa : 0.3045          
# Mcnemar's Test P-Value : 0.017706        
# 
# Sensitivity : 0.4815          
# Specificity : 0.8154          
# Pos Pred Value : 0.6842          
# Neg Pred Value : 0.6543          
# Prevalence : 0.4538          
# Detection Rate : 0.2185          
# Detection Prevalence : 0.3193          
# Balanced Accuracy : 0.6484          
# 
# 'Positive' Class : 0   


############### TASK3 ##############################################


dataPCA <- as.data.frame(pca$x) # This is the output data from prcomp analysis
head(dataPCA)

set.seed(20)
Clusters <- kmeans(dataPCA, centers=10, nstart=50)
Clusters

# So how well did the K-means clustering uncover the actual structure of the data contained in the MyGrade variable? A cross-tabulation of MyGrade and cluster membership is given by
confMat <- table(dataPCA$PC31, Clusters$cluster)
confMat
# We can quantify the agreement between MyGrade and cluster, using an adjusted Rank index provided by the flexclust package.
randIndex(confMat) # 0.1038752 for 10 clusters
# The adjusted Rand index provides a measure of the agreement between two partitions, adjusted for chance. It ranges from -1 (no agreement) to 1 (perfect agreement). 
set.seed(20)
ClustersFinal <- kmeans(dataPCA, 8, nstart=25)
confMatFinal <- table(dataPCA$PC31, ClustersFinal$cluster)
randIndex(confMatFinal) # 0.1685834  for 8 clusters (optimal cluster # from scree plot)

#####  Before ####
#### standalone code to plot scree plot for kmeans ###############
wssplot <- function(data, nc=20, seed=1234){ 
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i,nstart=1)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters", ylab="Variance explained", main="Before Dimension Reduction")}
wssplot(data) 
abline(v = 8, lty =2) 
############# kmeans scree plot done ###############################

###  Now ############
#### standalone code to plot scree plot for kmeans ###############
wssplot1 <- function(dataPCA, nc=20, seed=1234){  
  wss1 <- (nrow(dataPCA)-1)*sum(apply(dataPCA,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss1[i] <- sum(kmeans(dataPCA, centers=i,nstart=1)$withinss)}
  plot(1:nc, wss1, type="b", xlab="Number of Clusters", ylab="Variance explained", main="After Dimension Reduction")}
wssplot1(dataPCA) 
abline(v = 8, lty =2) 
############# kmeans scree plot done ###############################

# plot for report
par(mfrow=c(2,1))
wssplot(data) 
abline(v = 8, lty =2) 
wssplot1(dataPCA) 
abline(v = 8, lty =2) 
par(mfrow=c(1,1))


#### Expectation Maximization ##################
dataDefault1 <- mclustBIC(dataPCA)
dataCustomize1 <- mclustBIC(dataPCA, G = 1:20, x = dataDefault1)
plot(dataCustomize1, G = 3:20, legendArgs = list(x = "topright"),xlab='Number of clusters') # best value at 7th cluster
abline(v = 7, lty =2) 

############### TASK4 ##############################################

dataPCA <- as.data.frame(pca$x) # This is the output data from prcomp analysis
# split: .85, .8, .75, .7, .65, .6, .55, .5, .45
set.seed(200)
index.data <- sample(1:nrow(dataPCA),round(0.45*nrow(dataPCA)))
trainScaled <- dataPCA[index.data,] # 70%
testScaled <- dataPCA[-index.data,] # 30%
NN <- names(trainScaled)
ff <- as.formula(paste("PC31 ~", paste(NN[!NN %in% "PC31"], collapse = " + ")))
ff
# For learning curve, run the neural network for multiple times by changing the percentage of 
# train/test split and find MSE for each step. Plot that MSE in a graph.
# Calculating MSE.
NeuralNet <- neuralnet(ff,data=trainScaled,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Train_SSE <- sum((NeuralNet$net.result[[1]] - trainScaled[,31])^2)
MSE_train <- NN_Train_SSE/nrow(trainScaled)
MSE_train # Mean Squared Error  0.1350781249
nrow(trainScaled) # 276
listMSEtr=c(0.1482775817, 0.1171002659, 0.1394318619, 0.1350781249, 0.1082534565, 0.1131214381, 0.08753255714, 0.1022628185, 0.08153341087) # train data
listTr=c(336, 316, 296, 276, 257, 237, 217, 198, 178) # train data

NeuralNetTest <- neuralnet(ff,data=testScaled,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Test_SSE <- sum((NeuralNetTest$net.result[[1]]-testScaled[,31])^2)
MSE_test<-NN_Test_SSE/nrow(testScaled)
MSE_test
nrow(testScaled)
listMSEte=c(0.04407435405, 0.04463376316,   0.02589934399, 0.05542529601, 0.1145059221, 0.0902390286, 0.09282047811, 0.1031201844, 0.09279801747) # test data
listTe = c(59, 79, 99, 119, 138, 158, 178,197, 217) # test data

# plot for report
par(mfrow=c(2,1))
plot (listMSEte~listTe, type = "b", xlab = "Test data size", ylab = "Test Data MSE", main = "Learning curve for test dataset (School Data)",ylim=c(0,0.15))
plot (listMSEtr~listTr, type = "b", xlab = "Training data size", ylab = "Training Data MSE", main = "Learning curve for training dataset (School Data)",ylim=c(0,0.15))
par(mfrow=c(1,1))

############### TASK5 ##############################################
dataKmeans <- as.data.frame(ClustersFinal$centers) # This is the output data from kmeans with 8 clusters
# split: .85, .8, .75, .7, .65, .6, .55, .5, .45
set.seed(201)
index.data1 <- sample(1:nrow(dataKmeans),round(0.85*nrow(dataKmeans)))
trainScaled1 <- dataKmeans[index.data1,] # 70%
testScaled1 <- dataKmeans[-index.data1,] # 30%
# names(trainScaled1)
# dim(trainScaled1) # 7 31
NN1 <- names(trainScaled1)
ff1 <- as.formula(paste("MyGrade ~", paste(NN1[!NN1 %in% "MyGrade"], collapse = " + ")))
ff1
NeuralNet1 <- neuralnet(ff1,data=trainScaled1,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Train_SSE1 <- sum((NeuralNet1$net.result[[1]] - trainScaled1[,31])^2)
MSE_train1 <- NN_Train_SSE1/nrow(trainScaled1)
MSE_train1 # Mean Squared Error  0.1828376401
nrow(trainScaled1) # 7
listMSEtr1=c(0.1828376401,0.006122064112,0.006122064112,0.006122064112,0.1344578679,0.1344578679,0.1266985938,0.1266985938,0.1266985938) # train data
listTr1=c(7,6,6,6,5,5,4,4,4) # train data

NeuralNetTest1 <- neuralnet(ff1,data=testScaled1,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Test_SSE1 <- sum((NeuralNetTest1$net.result[[1]]-testScaled1[,31])^2)
MSE_test1<-NN_Test_SSE1/nrow(testScaled1)
MSE_test1
nrow(testScaled1)
listMSEte1=c(0.002035566436, 0.09332453108,0.09332453108,0.09332453108, 0.09794928906,0.09794928906,0.1716883991,0.1716883991, 0.1716883991) # test data
listTe1 = c(1,2,2,2,3,3,4,4,4)

# plot for report
par(mfrow=c(2,1))
plot (listMSEte1~listTe1, type = "b", xlab = "Test data cluster number", ylab = "Test Data MSE", main = "Learning curve for test dataset (School Data)",ylim=c(0,0.18))
plot (listMSEtr1~listTr1, type = "b", xlab = "Training data cluster number", ylab = "Training Data MSE", main = "Learning curve for training dataset (School Data)",ylim=c(0,0.18))
par(mfrow=c(1,1))


######################################################################
################### End of School Data ###############################
#################### Start of Bank Data ##############################
######################################################################
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
# Normalize bank.df
bank.df$Age <- normalize(bank.df$Age)
bank.df$Experience <- normalize(bank.df$Experience)
bank.df$Income <- normalize(bank.df$Income)
bank.df$Family <- normalize(bank.df$Family)
bank.df$CCAvg <- normalize(bank.df$CCAvg)
bank.df$Education <- normalize(bank.df$Education)
bank.df$Mortgage <- normalize(bank.df$Mortgage)
bank.df$SecuritiesAccount <- normalize(bank.df$SecuritiesAccount)
bank.df$CDAccount <- normalize(bank.df$CDAccount)
bank.df$Online <- normalize(bank.df$Online)
bank.df$CreditCard <- normalize(bank.df$CreditCard)
bank.df$PersonalLoan <- normalize(bank.df$PersonalLoan)

##################### TASK1 ##########################################
set.seed(21)
Clusters1 <- kmeans(bank.df, centers=10, nstart=50)
Clusters1

confMat1 <- table(bank.df$PersonalLoan, Clusters1$cluster)
confMat1
# We can quantify the agreement between MyGrade and cluster, using an adjusted Rank index provided by the flexclust package.
randIndex(confMat1) # 0.04010590441  for 10 clusters
# The adjusted Rand index provides a measure of the agreement between two partitions, adjusted for chance. It ranges from -1 (no agreement) to 1 (perfect agreement). 
set.seed(22)
ClustersFinal1 <- kmeans(bank.df, 6, nstart=25)
confMatFinal1 <- table(bank.df$PersonalLoan, ClustersFinal1$cluster)
randIndex(confMatFinal1) 
# 0.08745313475  for 6 clusters, this was the max value for clusters 1 t0 20
# Try to find higher rand index for better number of cluster by changing
# number of cluster and watching for higher rand index

dataKmeansBank <- as.data.frame(ClustersFinal1$centers) # This is the output data from kmeans with 8 clusters

#### standalone code to plot scree plot for kmeans ###############
wssBank <- function(bank.df, nc=50, seed=123){  # nc for number of centers
  wssBank <- (nrow(bank.df)-1)*sum(apply(bank.df,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wssBank[i] <- sum(kmeans(bank.df, centers=i,nstart=25)$withinss)}
  # For nstart>4 we did not see elbow, hence accepted R's default nstart=1 to observe elbow.
  plot(1:nc, wssBank, type="b", xlab="Number of Clusters", ylab="Variance explained")}
wssBank(bank.df) # best elbow plot, cluster number=8
abline(v = 6, lty =2) # draw a veritcal dashed line from 8th cluster
############# kmeans scree plot done ###############################

#### Expectation Maximization ##################
dataDefault1a <- mclustBIC(bank.df)
dataCustomize1a <- mclustBIC(bank.df, G = 1:50, x = dataDefault1a)
plot(dataCustomize1a, G = 1:50, legendArgs = list(x = "top",cex=.8),xlab='Number of clusters') # best value at 7th cluster
#abline(v = 7, lty =2) 

##################### TASK2 ##########################################
##### 1. For Decision Tree  use hclust (hierarchical clustering) ###

hcluster1 <- hclust(dist(bank.df), method='complete')
hcluster1 # this plots the cluster
plot(hcluster1)
abline(h = 2.6, lty =2) 
hclustTrim1 <- cutree(hcluster1,7)
plot(hclustTrim1)

Eucl_EX1<-vegdist(bank.df,"euclidean") 
Eucl1<-hclust(Eucl_EX1,method="complete") # The mothod is 'complete' linkage
# plot for report
plot(Eucl1,main='Complete Linkage Dendrogram',xlab='')
abline(h = 2.6, lty =2) 
bank.df$PersonalLoan
chord_EX1<-vegdist(decostand(bank.df,"norm"),"euclidean") 
chord_EX1
chord1<-hclust(chord_EX1,method="complete") 
plot(chord1)

# Cut tree so that it has 7 clusters
hclust.clusters1 <- cutree(chord1, k = 7)
plot(hclust.clusters1)


# Compare cluster membership to actual MyGrade
table(hclust.clusters1, bank.df$PersonalLoan)
sum(apply(table(hclust.clusters1, bank.df$PersonalLoan), 1, min)) # 203

# To compare with kmeans
abcd<-kmeans(scale(bank.df), centers = 7)
table(abcd$cluster,bank.df$PersonalLoan)

# Table for Report:
table(hclust.clusters1, bank.df$PersonalLoan)  # from hclust
table(abcd$cluster,bank.df$PersonalLoan)  # from kmeans


#################### 2. PCA using prcomp, and PCA for Investigate() utility ###########
pca1 <- prcomp(bank.df,scale=T) 
plot(pca1)
plot(pca1$sdev^2,xlab="Principal Components",ylab="Variance Explained")
plot(cumsum(pca1$sdev^2),xlab="Principal Components",ylab="Cumulative Variance Explained")

# A few plots for report from this:
PCA2 <- PCA(bank.df,ncp=10,graph=T, scale.=T) 
Investigate(PCA2) 

######## output data from PCA ##############
bank.dfPCA <- as.data.frame(pca1$x) # This is the output data from prcomp analysis
head(bank.dfPCA[,1:2])
plot(PC1~PC2, data=bank.dfPCA, cex = 0.5, lty = "solid")
#text(PC1~PC2, data=bank.dfPCA, labels=rownames(bank.df),cex=.8)
##############################################

biplot(pca1,scale=0) # stick with scale=0
pcavar1<- pca1$sdev^2
pve1<-pcavar1/sum(pcavar1)
ggbiplot(pca1,circle=T) # shows only dots

colMeans(bank.df) # Mean of each variable
apply(bank.df,2,sd) # Standard deviation of each variable

# Plot for report
par(mfrow=c(2,1))
plot(pve1,xlab='Principal Components', ylab='Proportion of variance explained',ylim=c(0.01,0.2)) # this is like elbow plot (elbow at 7)
abline(v = 4, lty =2)
plot(cumsum(pve1),xlab='Principal Components', ylab='Cumulative Proportion of variance explained',ylim=c(0.01,1)) # this is like elbow plot (elbow at 7)
abline(v = 4, lty =2)
abline(h = 0.56, lty =2)
par(mfrow=c(1,1))

# The total of 42.961% variance explained by 7 principal components

############## ICA  ############

icaModel1 <- fastICA(bank.df, 4, alg.typ = "deflation", fun = "logcosh", alpha = 1,
                     method = "R", row.norm = FALSE, maxit = 200,
                     tol = 0.0001, verbose = F)

# plot for report
par(mfrow = c(1, 3))
plot(icaModel1$X, main = "Original Component",xlim=c(-2,2),ylim=c(-2.2,2))
plot(icaModel1$X %*% icaModel1$K, main = "PCA components",xlim=c(-2,2),ylim=c(-2.2,2))
plot(icaModel1$S, main = "ICA components",xlim=c(-2,2),ylim=c(-2.2,2))
par(mfrow = c(1, 1))

names(icaModel1)
summary(icaModel1)
icaModel1$X # pre-processed data matrix 
icaModel1$K # pre-whitening matrix that projects data onto the first n.comp principal components.
icaModel1$W # estimated un-mixing matrix  (no.factors by no.signals)  - no. means number of 
icaModel1$A # estimated mixing matrix (no.signals by no.factors) - no. means number of 
icaModel1$S # estimated source matrix (The column vectors of estimated independent components (no.obs by no.factors)) - here no.factors means number of principal components

# check whitening:
# check correlations are zero
cor(icaModel1$K) # correlations are not zero
# check diagonals are 1 in covariance
cov(icaModel1$K) # diagonals are not 1

cor(icaModel1$X) 
cor(icaModel1$W) 
#cor(icaModel$A)

# table for report 
# no correlations as off-diagonal elements are zero, good.
cor(icaModel1$S) 
cov(icaModel1$S)  
# diagonal elements are 1, good

######## Randomized Projections (needs more than 2 hours of runtime) #####
# https://cran.r-project.org/web/packages/RandPro/RandPro.pdf
set.seed(101)
index.data1a <- sample(1:nrow(bank.df),round(0.70*nrow(bank.df)))
Train1 <- bank.df[index.data1a,] # 70%
Test1 <- bank.df[-index.data1a,] # 30%
trainla<-as.factor(Train1$PersonalLoan) # need to declare as factor.
testla<-as.factor(Test1$PersonalLoan)  # need to declare as factor.
train1<-Train1[,1:11]
test1<-Test1[,1:11]
randomProjection<-classify(train1, test1, trainla, testla)
# 
# Confusion Matrix and Statistics
# Reference
# Prediction    0    1
# 0 1354   58
# 1    7   81
# 
# Accuracy : 0.9567          
# 95% CI : (0.9451, 0.9664)
# No Information Rate : 0.9073          
# P-Value [Acc > NIR] : 2.595e-13       
# 
# Kappa : 0.6915          
# Mcnemar's Test P-Value : 5.584e-10       
# 
# Sensitivity : 0.9949          
# Specificity : 0.5827          
# Pos Pred Value : 0.9589          
# Neg Pred Value : 0.9205          
# Prevalence : 0.9073          
# Detection Rate : 0.9027          
# Detection Prevalence : 0.9413          
# Balanced Accuracy : 0.7888          
# 
# 'Positive' Class : 0               


##################### TASK3 ##########################################
######## output data from PCA ##############
bank.dfPCA <- as.data.frame(pca1$x) # This is the output data from prcomp analysis
head(bank.dfPCA)
##############################################

set.seed(20)
Clusters1a <- kmeans(bank.dfPCA, centers=10, nstart=50)
Clusters1a

# So how well did the K-means clustering uncover the actual structure of the data contained in the MyGrade variable? A cross-tabulation of MyGrade and cluster membership is given by
confMat1a <- table(bank.dfPCA$PC12, Clusters1a$cluster)
confMat1a
# We can quantify the agreement between MyGrade and cluster, using an adjusted Rank index provided by the flexclust package.
randIndex(confMat1a) # 1.55073e-05  for 10 clusters
# The adjusted Rand index provides a measure of the agreement between two partitions, adjusted for chance. It ranges from -1 (no agreement) to 1 (perfect agreement). 
set.seed(20)
ClustersFinal1a <- kmeans(bank.dfPCA, 6, nstart=25)
confMatFinal1a <- table(bank.dfPCA$PC12, ClustersFinal1a$cluster)
randIndex(confMatFinal1a) # 5.847072e-06  for 6 clusters (optimal cluster # from scree plot)

#####  Before ####
#### standalone code to plot scree plot for kmeans ###############
wssplot1a <- function(bank.df, nc=20, seed=124){ 
  wss1a <- (nrow(bank.df)-1)*sum(apply(bank.df,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss1a[i] <- sum(kmeans(bank.df, centers=i,nstart=1)$withinss)}
  plot(1:nc, wss1a, type="b", xlab="Number of Clusters", ylab="Variance explained", main="Before Dimension Reduction")}
wssplot1a(bank.df) 
abline(v = 6, lty =2) 
############# kmeans scree plot done ###############################

###  Now ############
#### standalone code to plot scree plot for kmeans ###############
wssplot1b <- function(bank.dfPCA, nc=20, seed=134){  
  wss1b <- (nrow(bank.dfPCA)-1)*sum(apply(bank.dfPCA,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss1b[i] <- sum(kmeans(bank.dfPCA, centers=i,nstart=1)$withinss)}
  plot(1:nc, wss1b, type="b", xlab="Number of Clusters", ylab="Variance explained", main="After Dimension Reduction")}
wssplot1b(bank.dfPCA) 
abline(v = 6, lty =2) 
############# kmeans scree plot done ###############################

# plot for report
par(mfrow=c(2,1))
wssplot1a(bank.df) 
abline(v = 6, lty =2) 
wssplot1b(bank.dfPCA) 
abline(v = 6, lty =2) 
par(mfrow=c(1,1))

#### Expectation Maximization ##################

# dataDefault1a <- mclustBIC(bank.df)
# dataCustomize1a <- mclustBIC(bank.df, G = 1:50, x = dataDefault1a)
# plot(dataCustomize1a, G = 1:50, legendArgs = list(x = "top",cex=.8),xlab='Number of clusters') # best value at 7th cluster
Default1b <- mclustBIC(bank.dfPCA)
Customize1b <- mclustBIC(bank.dfPCA, G = 1:50, x = Default1b)
plot(Customize1b, G = 1:20, legendArgs = list(x = "top",cex=.8),xlab='Number of clusters', main="After Dimension Reduction") # best value at 7th cluster
abline(v = 6, lty =2) 

# plot for report
par(mfrow=c(1,2))
plot(dataCustomize1a, G = 1:20, legendArgs = list(x = "top",cex = .8),xlab='Number of clusters')
abline(v = 6, lty =2) 
plot(Customize1b, G = 1:20,legendArgs = list(x = "top",cex = .8) ,xlab='Number of clusters') 
abline(v = 6, lty =2) 
par(mfrow=c(1,1))

##################### TASK4 ##########################################

bank.dfPCA <- as.data.frame(pca1$x) # This is the output data from prcomp analysis

# split: .85, .8, .75, .7, .65, .6, .55, .5, .45
set.seed(201)
index.data2 <- sample(1:nrow(bank.dfPCA),round(0.45*nrow(bank.dfPCA)))
trainScaled2 <- bank.dfPCA[index.data2,] # 70%
testScaled2 <- bank.dfPCA[-index.data2,] # 30%
NN2 <- names(trainScaled2)
ff2 <- as.formula(paste("PC12 ~", paste(NN2[!NN2 %in% "PC12"], collapse = " + ")))
ff2
NeuralNet2 <- neuralnet(ff2,data=trainScaled2,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Train_SSE2 <- sum((NeuralNet2$net.result[[1]] - trainScaled2[,12])^2)
MSE_train2 <- NN_Train_SSE2/nrow(trainScaled2)
MSE_train2 # Mean Squared Error 0.005128815563
nrow(trainScaled2) # 4250
listMSEtr2=c(0.005128815563, 0.005080296713, 0.005095116637, 0.00529211685, 0.004964216109, 0.005154666754, 0.005104140276, 0.005215575954, 0.004915072731) # train data
listTr2=c(4250, 4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250) # train data

NeuralNetTest2a <- neuralnet(ff2,data=testScaled2,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Test_SSE2a <- sum((NeuralNetTest2a$net.result[[1]]-testScaled2[,12])^2)
MSE_test2a<-NN_Test_SSE2a/nrow(testScaled2)
MSE_test2a
nrow(testScaled2)
listMSEte2=c( 0.004891277009, 0.004930405796, 0.00510313953, 0.004967961858, 0.005117855377, 0.0051708747, 0.004934932701, 0.005154444834,  0.005108289679) # test data
listTe2 = c(750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750) # test data

# plot for report
par(mfrow=c(2,1))
plot (listMSEte2~listTe2, type = "b", xlab = "Test data size", ylab = "Test Data MSE", main = "Learning curve for test dataset (Bank Data)",ylim=c(0.0046,0.0053))
plot (listMSEtr2~listTr2, type = "b", xlab = "Training data size", ylab = "Training Data MSE", main = "Learning curve for training dataset (Bank Data)",ylim=c(0.0046,0.0053))
par(mfrow=c(1,1))

##################### TASK5 ##########################################
dataKmeansBank <- as.data.frame(ClustersFinal1$centers)  # This is the output data from kmeans with 8 clusters
# split: .85, .8, .75, .7, .65, .6, .55, .5, .45
set.seed(201)
index.data3 <- sample(1:nrow(dataKmeansBank),round(0.85*nrow(dataKmeansBank)))
trainScaled3 <- dataKmeansBank[index.data3,] 
testScaled3 <- dataKmeansBank[-index.data3,] 
NN3 <- names(trainScaled3)
ff3 <- as.formula(paste("PersonalLoan ~", paste(NN3[!NN3 %in% "PersonalLoan"], collapse = " + ")))
ff3
NeuralNet3 <- neuralnet(ff3,data=trainScaled3,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Train_SSE3 <- sum((NeuralNet3$net.result[[1]] - trainScaled3[,12])^2)
MSE_train3 <- NN_Train_SSE3/nrow(trainScaled3)
MSE_train3 # Mean Squared Error  0.1423780342
nrow(trainScaled3) # 5
listMSEtr3=c(0.1423780342, 0.1423780342, 0.00651839908, 0.00651839908, 0.00651839908, 0.00651839908, 0.2052435674, 0.2052435674, 0.2052435674) # train data
listTr3=c(5, 5, 4, 4, 4, 4, 3, 3, 3) # train data

NeuralNetTest3 <- neuralnet(ff3,data=testScaled3,hidden=c(3,2), threshold = 0.05, act.fct='logistic')
NN_Test_SSE3 <- sum((NeuralNetTest3$net.result[[1]]-testScaled3[,12])^2)
MSE_test3<-NN_Test_SSE3/nrow(testScaled3)
MSE_test3
nrow(testScaled3)
listMSEte3=c(0.001485082381, 0.001485082381, 0.00002609464794, 0.00002609464794, 0.00002609464794, 0.00002609464794, 0.000008424406525, 0.000008424406525, 0.000008424406525) # test data
listTe3 = c(1, 1, 2, 2, 2, 2, 3, 3, 3)

# plot for report
par(mfrow=c(2,1))
plot (listMSEte3~listTe3, type = "b", xlab = "Test data cluster number", ylab = "Test Data MSE", main = "Learning curve for test dataset (Bank Data)",ylim=c(0,0.0018))
plot (listMSEtr3~listTr3, type = "b", xlab = "Training data cluster number", ylab = "Training Data MSE", main = "Learning curve for training dataset (Bank Data)",ylim=c(0,0.22))
par(mfrow=c(1,1))


################### End ##################












