################################################################
# Ramesh Subedi

# Data description here: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Data from here: https://archive.ics.uci.edu/ml/machine-learning-databases/00320/


# Perform linear and logistic regression on the given dataset. 
# In addition, experiment with design and feature choices.

# Tasks:
# 1. Divide the dataset into train and test sets sampling randomly. 
# Use only predictive attributes and the target variable (do not use non-predictive attributes). 
# Also, do not use G1 and G2.
# 2. Use linear regression to predict the final grade (G3). Report and compare your train and 
# test error/accuracy metrics. You can pick any metrics you like 
# (e.g. mean squared error, mean absolute error, etc.).
# 3. Convert this problem into a binary classification problem. The target variable should 
# have two grade categories.
# 4. Implement logistic regression to carry out classification on this data set. Report 
# accuracy/error metrics for train and test sets.
# 
# Experimentation:
# 1. Experiment with various model parameters for both linear and logistic regression and 
# report on your findings as how the error varies for train and test sets with varying these 
# parameters. Plot the results. Report your best parameters. Examples of these parameters can 
# be learning rate for gradient descent, convergence threshold, etc.
# 2. Pick ten features randomly and retrain your model only on these ten features. Compare 
# train and test error results for the case of using all features to using ten random features. 
# Report which ten features did you select randomly.
# 3. Now pick ten features that you think are best suited to predict the output, and retrain your model using these ten features. Compare to the case of using all features and to random features case. Did your choice of features provide better results than picking random features? Why? Did your choice of features provide better results than using all features? Why?

###############################################################

rm(list=ls()) #drop all variables

######################################
library(data.table) 
library(magrittr)
library(dtplyr) 
library(sandwich) # for White correction
library(lmtest) # for more advanced hypothesis testing tools
library(DBI) 
library(RSQLite) 
library(tidyverse)
library(broom)  # for tidy() function
#library(TSA)
#library(forecast)
#library(vars)
#library(fpp) # for VAR forecast
#library(UsingR)
#library(margins)
#library(plm) # for pooled OLS (ordinary least squares)
library(car) # for scatterplot()
#library(aod) # for probit link
library(gradDescent) # for Gradient Descent calculation
#library(glmnet)

#sessionInfo()
#RStudio.Version()
# search() # displays all packages currently being used

problemData <- read.table("~/mlData/data1/student-mat.csv",sep=";",header=TRUE)
names(problemData)


################ TASK 1 #######################

# split whole problemData data into 70% for training and 30% for testing.

set.seed(1) # set fixed seed so that radom sampling for splitting data in 80/20 ratio is reproducible.

# Randomly sample 80% of the row IDs for training
train.rows <- sample(rownames(problemData), dim(problemData)[1]*0.8)

# assign the remaining 20% row IDs serve as test
test.rows <- sample(setdiff(rownames(problemData), train.rows),dim(problemData)[1]*0.2)

# create the 3 data frames by collecting all columns from the appropriate rows
train.data <- problemData[train.rows, ]
test.data <-  problemData[test.rows, ]

class(train.data)
names(train.data)
train.data$school

train.data <- train.data %>% dplyr::select(-G1,-G2) # Drop G2 and G2 variables from train.data

# Factor Variables: school,sex,address,famsize,Pstatus,Fjob,Mjob,reason,guardian,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic

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


lm(G3~., data=train.data)%>% step%>%summary 

# The best model found from above using step function is the following (with lowest AIC=913.5) for 80% data:




################ TASK 2 #######################

# With 80% data
model_train <- lm(G3~sex+age+famsize+reason+traveltime+studytime+failures+schoolsup+romantic+goout+absences, data = train.data)

# With 70% data
#model_train <- lm(G3 ~ sex + famsize + reason + traveltime + failures + schoolsup + higher + internet + romantic + goout, data = train.data)

summary(model_train)
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 18.32442    3.98153   4.602 6.14e-06 ***
# sex          1.03070    0.50556   2.039   0.0423 *  
# age         -0.44127    0.21318  -2.070   0.0393 *  
# famsize      1.06647    0.52303   2.039   0.0423 *  
# reason       0.42321    0.19762   2.142   0.0330 *  
# traveltime  -0.58881    0.34391  -1.712   0.0879 .  
# studytime    0.52323    0.30035   1.742   0.0825 .  
# failures    -1.66085    0.35122  -4.729 3.46e-06 ***
# schoolsup   -1.79807    0.75738  -2.374   0.0182 *  
# romantic    -0.92527    0.51462  -1.798   0.0732 .  
# goout       -0.31905    0.21834  -1.461   0.1450    
# absences     0.04580    0.03113   1.471   0.1422    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.165 on 304 degrees of freedom
# Multiple R-squared:  0.2006,	Adjusted R-squared:  0.1717 
# F-statistic: 6.935 on 11 and 304 DF,  p-value: 1.849e-10


# names(model) # find out which metrics are available
# plot(model$coefficients)

plot(model_train$residuals)
plot(model_train$residuals^2)
plot(predict(model_train))
mse<-mean(model_train$residuals^2) # mean squared error (MSE)
mse # 16.69132

# No need of this block (this block has same info as above)
reg.model<-summary(model_train)
names(reg.model)
#class(reg.model)]
plot(reg.model$residuals)
plot(reg.model$residuals^2)
mse <- mean(reg.model$residuals^2)
mse  ## mean squared error 16.69132


# This result from step function appears to be the same whether you scale the data or not. Or, the scale is not changing anything in our problem.


# Now change the factors into numeric in test set also and repeat above process. And see if
# the adjusted R-squared turns out to be the same for test data as for training data.

test.data <- test.data %>% dplyr::select(-G1,-G2) # Drop G2 and G2 variables from test.data
# Factor variables: school,sex,address,famsize,Pstatus,Fjob,Mjob,reason,guardian,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic

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

test.data%>%mutate_if(is.numeric, scale) # This scaling works. Though scaling is only for numeric variables (not for factors), we changed factors into numeric above. Hence this scaling works for all variables.

names(test.data)
class(test.data) # check if test.data is still data.frame

# Now use the same variables used in linear regression using train.data for test.data as well
# so that we can compare Adjusted R-squared in both cases.
# With 20% test data
model_test <- lm(G3~sex+age+famsize+reason+traveltime+studytime+failures+schoolsup+romantic+goout+absences, data = test.data)

summary(model_test)
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -9.7134 -1.6106  0.3373  1.8245 10.8424 
# 
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 13.139166   8.115903   1.619  0.11016    
# sex          1.853075   1.048383   1.768  0.08169 .  
# age         -0.174863   0.369200  -0.474  0.63731    
# famsize      0.066734   1.095261   0.061  0.95160    
# reason       0.092827   0.451121   0.206  0.83760    
# traveltime  -0.001131   0.715941  -0.002  0.99874    
# studytime    0.094164   0.677305   0.139  0.88985    
# failures    -2.313675   0.612585  -3.777  0.00034 ***
# schoolsup   -0.373805   1.411641  -0.265  0.79198    
# romantic    -0.998564   1.079193  -0.925  0.35814    
# goout       -0.368197   0.444979  -0.827  0.41092    
# absences     0.081421   0.058728   1.386  0.17022    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.201 on 67 degrees of freedom
# Multiple R-squared:  0.2798,	Adjusted R-squared:  0.1615 
# F-statistic: 2.366 on 11 and 67 DF,  p-value: 0.0155

reg.model<-summary(model_test)
#class(reg.model)
names(reg.model)
plot(reg.model$residuals)
plot(reg.model$residuals^2)
mse <- mean(reg.model$residuals^2)
mse  ## mean squared error  14.9651

# Even though both R-squared and adjusted R-squared have gone up for test data
# compared to the training data, most of the coefficients of the features 
# in both data are not significant.
# 
#  R-squaredTraingData = 0.2006
#  R-squaredTestData   = 0.2798
# 
#  Adjusted R-squaredTrainingData:  0.1717  
#  Adjusted R-squaredTestData:      0.1615 
# Unlike training data features, most of the coefficients of test data features are not significant.
#par(mfrow=c(2,1))
plotLMtrain <- ggplot(train.data,aes(x=sex+age+famsize+reason+traveltime+studytime+failures+schoolsup+romantic+goout+absences,y=G3)) + geom_point()
#plotLMtrain
plotLMtrain+xlim(25,105)+ylim(0,21)+geom_point(aes(y=predict(model_train)),color="red",size=1) +labs(x="Features",y="Final Grade G3")

plotLMtest <- ggplot(test.data,aes(x=sex+age+famsize+reason+traveltime+studytime+failures+schoolsup+romantic+goout+absences,y=G3)) + geom_point()

plotLMtest+xlim(25,105)+ylim(0,21)+geom_point(aes(y=predict(model_test)),color="red",size=1)+labs(x="Features",y="Final Grade G3")
#par(mfrow=c(1,1))



################ TASK 3 #######################

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
summary(xdata)
xdata%>%mutate_if(is.numeric, scale) # scales the numeric data, leaves non-numeric alone.



################ TASK 4 #######################
# dim(train.data) # 276 records, 31 variables
# sum(is.na(train.data$G3)) # everyone has G3 value, no empty data for G3.

# now use myGrade instead of G3 for the following logistic regression
glm(myGrade~., family=binomial,data=xdata)%>% step%>%summary  # The step function selects the best model
# This is the best model found by the step function.
model_glm <- glm(myGrade~sex+age+famsize+Fedu+reason+traveltime+failures+schoolsup+activities+nursery+higher+internet+Walc,family=binomial,data=xdata)
summary(model_glm)

# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.0143  -1.0575   0.5488   0.9447   2.3891  
# 
# Coefficients:
#              Estimate Std. Error z value Pr(>|z|)   
# (Intercept)   2.9050     2.9543   0.983  0.32545   
# sex           0.4614     0.2749   1.678  0.09332 . 
# age          -0.2223     0.1180  -1.884  0.05957 . 
# famsize       0.4184     0.2862   1.462  0.14377   
# Fedu          0.2040     0.1251   1.631  0.10289   
# reason        0.2168     0.1066   2.034  0.04197 * 
# traveltime   -0.3751     0.1975  -1.899  0.05762 . 
# failures     -0.7963     0.2504  -3.180  0.00147 **
# schoolsup    -1.2462     0.4156  -2.999  0.00271 **
# activities   -0.4044     0.2672  -1.513  0.13019   
# nursery      -0.5933     0.3443  -1.723  0.08486 . 
# higher        1.0462     0.7378   1.418  0.15622   
# internet      0.5691     0.3455   1.647  0.09950 . 
# Walc         -0.2558     0.1124  -2.275  0.02291 * 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 437.04  on 315  degrees of freedom
# Residual deviance: 366.96  on 302  degrees of freedom
# AIC: 394.96
# 
# Number of Fisher Scoring iterations: 5

reg.modelGLM <- summary(model_glm)
class(reg.modelGLM)
names(reg.modelGLM)
plot(reg.modelGLM$deviance.resid)
plot(reg.modelGLM$deviance.resid^2)
mse <- mean(reg.modelGLM$deviance.resid^2)
mse # 1.16128, this is MSE (Mean Squared Error)

# Alternative to get MSE
plot(resid(model_glm))
plot(resid(model_glm)^2)
mse <- mean(resid(model_glm)^2)
mse #   MSE = 1.16128

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


model_glmTestData <- glm(myGrade1~sex+age+famsize+Fedu+reason+traveltime+failures+schoolsup+activities+nursery+higher+internet+Walc,family=binomial,data=xdata1)
summary(model_glmTestData)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.8333  -0.9301   0.5549   0.8301   2.3001  
# 
# Coefficients:
#              Estimate Std. Error z value Pr(>|z|)   
# (Intercept)  3.38195    6.00809   0.563  0.57350   
# sex          0.41592    0.61199   0.680  0.49675   
# age         -0.22860    0.22089  -1.035  0.30071   
# famsize      0.18214    0.63137   0.288  0.77298   
# Fedu        -0.07842    0.28349  -0.277  0.78207   
# reason      -0.05286    0.24884  -0.212  0.83179   
# traveltime   0.49837    0.46312   1.076  0.28188   
# failures    -1.23942    0.41960  -2.954  0.00314 **
# schoolsup   -0.22572    0.77509  -0.291  0.77089   
# activities  -0.18095    0.57386  -0.315  0.75252   
# nursery     -0.87054    0.69062  -1.261  0.20749   
# higher       1.71472    1.39360   1.230  0.21854   
# internet    -0.62820    0.79392  -0.791  0.42879   
# Walc         0.11752    0.21943   0.536  0.59224   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 102.72  on 78  degrees of freedom
# Residual deviance:  85.58  on 65  degrees of freedom
# AIC: 113.58
# 
# Number of Fisher Scoring iterations: 4

reg.modelGLM1 <- summary(model_glmTestData)
class(reg.modelGLM1)
names(reg.modelGLM1)
plot(reg.modelGLM1$deviance.resid)
plot(reg.modelGLM1$deviance.resid^2)
mse1 <- mean(reg.modelGLM1$deviance.resid^2)
mse1 # 1.083289, this is MSE (Mean Squared Error) for test.data

#par(mfrow=c(2,1))
plotGLMtrain <- ggplot(xdata,aes(x=sex+age+famsize+Fedu+reason+traveltime+failures+schoolsup+activities+nursery+higher+internet+Walc,y=myGrade)) + geom_point()
plotGLMtrain
plotGLMtrain+xlim(27,47)+ylim(-5,2.7)+geom_point(aes(y=predict(model_glm)),color="red",size=1) +labs(x="Features",y="Final Grade G3")

plotGLMtest <- ggplot(xdata1,aes(x=sex+age+famsize+Fedu+reason+traveltime+failures+schoolsup+activities+nursery+higher+internet+Walc,y=myGrade1)) + geom_point()
plotGLMtest
plotGLMtest+xlim(27,47)+ylim(-5,2.5)+geom_point(aes(y=predict(model_glmTestData)),color="red",size=1)+labs(x="Features",y="Final Grade G3")
par(mfrow=c(1,1))

####################### End of TASK Part ###############

############### Start of Experimentation ###############
################### Experimentation Part 1 #############

# Plot a graph for how MSE veries with alpha in gradient descent.

# Just to check if the features are still numeric
names(train.data) 
train.data%>%mutate_if(is.numeric, scale) # yes they are
names(test.data) 
test.data%>%mutate_if(is.numeric, scale) # yes they are
class(train.data)

X1 <- model.matrix(G3~.,data=train.data)
y1 <- train.data$G3
fit_ridge_cv1 <- cv.glmnet(X1, y1, alpha = 0.0) 
# cv.glmnet means Cross-Validation For Glmnet
plot(fit_ridge_cv1)
glance(fit_ridge_cv1)

# Lamda min for train.data is 4.756173 for minimum MSE (which is little below 19)

X2 <- model.matrix(G3~.,data=test.data)
y2 <- test.data$G3
fit_ridge_cv2 <- cv.glmnet(X2, y2, alpha = 0.0) 
plot(fit_ridge_cv2)
glance(fit_ridge_cv2) # lamda min = 2.076146

# Lamda min for test.data is 2.076146 for minimum MSE (which is little below 18)


# For logistic do this way, but make sure that y is binary.
#fit_1se <- glmnet(X, y, family = "binomial", lambda = 0.003)

X3 <- model.matrix(myGrade~.,data=xdata)
y3 <- myGrade
fit_cv3 <- cv.glmnet(X3, y3, family = "binomial", alpha = 0) 
warnings()
coef(fit_cv3)
plot(fit_cv3)
glance(fit_cv3) # lambda min = 0.2142894

# Lamda min for train.data is 0.2142894 for minimum MSE (which is little below 1.3)
# The MSE in this case is Binomial Deviance


X4 <- model.matrix(myGrade1~.,data=xdata1)
y4 <- myGrade1
#y4
fit_cv4 <- cv.glmnet(X4, y4, family = "binomial", alpha = 0) 
coef(fit_cv4)
plot(fit_cv4)
glance(fit_cv4) # lambda min = 119.0678

# Lamda min for test.data is 119.0678 for minimum MSE (which is little below 1.4)
# The MSE in this case is Binomial Deviance


# Plots for Experimentation Part1:
par(mfrow=c(2,2))
plot(fit_ridge_cv1)
plot(fit_ridge_cv2)
plot(fit_cv3)
plot(fit_cv4)
par(mfrow=c(1,1))

################### Experimentation Part 2 #############

# Now picking 10 features randomly in train.data:

dim(train.data) # 276 records, 31 variables
sum(is.na(train.data$G3)) # everyone has G3 value, no empty data for G3.


library(leaps)
regfitFull<-regsubsets(G3~.,data=train.data,nvmax=30)
summary(regfitFull)
reg.summary<-summary(regfitFull)
#class(reg.summary)
names(reg.summary)
coef(regfitFull,10) # This gives names of 10 best variables.
#reg.summary$obj
#plot(reg.summary$obj) # shows the plot of all 30 variables with their names on x axis

# coef(regfitFull,10)
# (Intercept)         sex         age     famsize        Medu      reason   studytime    failures   schoolsup 
# 15.4745164   0.8267408  -0.3524240   1.0443099   0.4083546   0.4174709   0.4944758  -1.5557752  -1.7520761 
# romantic       goout 
# -0.9145147  -0.3415725 
reg.summary$rss
reg.summary$bic
plot(reg.summary$rss)
plot(reg.summary$rsq)
mean(reg.summary$rss)
mean(reg.summary$rsq) # 0.2036727

par(mfrow=c(2,1))
plot(reg.summary$rsq,xlab='Features',ylab='R-squared')
plot(reg.summary$adjr2,xlab='Features',ylab='Adjusted R-squared')
par(mfrow=c(1,1))

# Retraining the model with these new randomly picked 10 variables for train.data:
names(train.data)
# I had to redo the factor to numeric conversion of train.data to carry out the following  regression.
model_10features <- lm(G3~sex+age+famsize+Medu+reason+studytime+failures+schoolsup+romantic+goout,data=train.data)
summary(model_10features)
mean(reg.summary$rsq)
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -11.8129  -1.7713   0.3872   2.7910   9.2715 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  15.4745     4.0701   3.802 0.000173 ***
# sex           0.8267     0.5095   1.623 0.105721    
# age          -0.3524     0.2125  -1.658 0.098291 .  
# famsize       1.0443     0.5224   1.999 0.046491 *  
# Medu          0.4084     0.2311   1.767 0.078284 .  
# reason        0.4175     0.1986   2.102 0.036368 *  
# studytime     0.4945     0.2976   1.661 0.097673 .  
# failures     -1.5558     0.3584  -4.341 1.93e-05 ***
# schoolsup    -1.7521     0.7574  -2.313 0.021371 *  
# romantic     -0.9145     0.5123  -1.785 0.075255 .  
# goout        -0.3416     0.2196  -1.555 0.120938    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.172 on 305 degrees of freedom
# Multiple R-squared:  0.1954,	Adjusted R-squared:  0.169 
# F-statistic: 7.407 on 10 and 305 DF,  p-value: 1.561e-10

reg.model<-summary(model_10features)
mse <- mean(reg.model$residuals^2)
mse  ## mean squared error  16.80002 (before retrain mse was 16.69132)


# Retraining the model with these new randomly picked 10 variables for test.data:
# I had to redo the factor to numeric conversion of train.data to carry out the following  regression.
model_10features1 <- lm(G3~sex+age+famsize+Medu+reason+studytime+failures+schoolsup+romantic+goout,data=test.data)
summary(model_10features1)

# Residuals:
#   Min       1Q   Median       3Q      Max 
# -11.6141  -1.7567   0.2543   1.9350   8.8613 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)   
# (Intercept)  8.05511    7.87830   1.022   0.3102   
# sex          1.95462    1.01920   1.918   0.0593 . 
# age         -0.04447    0.35793  -0.124   0.9015   
# famsize      0.40382    1.07020   0.377   0.7071   
# Medu         1.02273    0.45439   2.251   0.0276 * 
# reason       0.25290    0.43260   0.585   0.5608   
# studytime    0.13253    0.65300   0.203   0.8398   
# failures    -1.97974    0.60574  -3.268   0.0017 **
# schoolsup    0.16582    1.35529   0.122   0.9030   
# romantic    -1.33176    1.06342  -1.252   0.2147   
# goout       -0.55654    0.44091  -1.262   0.2112   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.081 on 68 degrees of freedom
# Multiple R-squared:  0.3101,	Adjusted R-squared:  0.2087 
# F-statistic: 3.057 on 10 and 68 DF,  p-value: 0.002945

reg.model1<-summary(model_10features1)
mse1 <- mean(reg.model1$residuals^2)
mse1  ## mean squared error 14.33447  (before retrain: mse was 14.9651) 


# Retraining for logistic model with train.data


# I had to re-run the G3 catagorizing process before doing the following  regression for train.data. 
grade <-train.data$G3
grade
meanVal<-mean(train.data$G3)
meanVal
myGrade <- ifelse(grade>=meanVal,1,0)
myGrade
grade
plot(myGrade,xlab='Student Number',ylab='Final Grade (G3)')

xdata <- train.data %>% dplyr::select(-G3) # Drop G3 since it's new name is grade.
summary(xdata)
xdata%>%mutate_if(is.numeric, scale) # scales the numeric data, leaves non-numeric alone.

model_glmRe <- glm(myGrade~sex+age+famsize+Medu+reason+studytime+failures+schoolsup+romantic+goout,family=binomial,data=xdata)
summary(model_glmRe)

# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.8323  -1.1137   0.6169   0.9979   1.9911  
# 
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  4.46448    2.20512   2.025 0.042908 *  
# sex          0.35208    0.26887   1.309 0.190370    
# age         -0.25202    0.11595  -2.174 0.029734 *  
# famsize      0.31059    0.27367   1.135 0.256404    
# Medu         0.10114    0.11972   0.845 0.398217    
# reason       0.16464    0.10373   1.587 0.112471    
# studytime    0.23575    0.15826   1.490 0.136330    
# failures    -0.86209    0.24889  -3.464 0.000533 ***
# schoolsup   -1.21770    0.40455  -3.010 0.002612 ** 
# romantic     0.02695    0.26858   0.100 0.920068    
# goout       -0.20978    0.11642  -1.802 0.071563 .  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 437.04  on 315  degrees of freedom
# Residual deviance: 382.79  on 305  degrees of freedom
# AIC: 404.79
# 
# Number of Fisher Scoring iterations: 4

mse_retrain <- mean(resid(model_glmRe)^2)
mse_retrain # MSE = 1.211348. It was 1.16128 before retraining.

# Retraining the glm model with test.data
# I had to re-run the G3 catagorizing process for test.data before doing the following regression.
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

model_glmTe <- glm(myGrade1~sex+age+famsize+Medu+reason+studytime+failures+schoolsup+romantic+goout,family=binomial,data=xdata1)
summary(model_glmTe)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.9718  -1.0603   0.5993   0.8316   1.8752  
# 
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)  
# (Intercept)   4.0494     4.4981   0.900   0.3680  
# sex           0.4110     0.5973   0.688   0.4914  
# age          -0.1832     0.2128  -0.861   0.3892  
# famsize       0.2577     0.6252   0.412   0.6802  
# Medu          0.3326     0.2702   1.231   0.2183  
# reason       -0.0321     0.2439  -0.132   0.8953  
# studytime     0.2484     0.3866   0.642   0.5206  
# failures     -0.8127     0.3669  -2.215   0.0268 *
# schoolsup    -0.2424     0.7733  -0.313   0.7539  
# romantic     -0.4786     0.6286  -0.761   0.4465  
# goout        -0.3859     0.2611  -1.478   0.1394  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 102.723  on 78  degrees of freedom
# Residual deviance:  86.748  on 68  degrees of freedom
# AIC: 108.75
# 
# Number of Fisher Scoring iterations: 4
mse_retrainTest <- mean(resid(model_glmTe)^2)
mse_retrainTest # MSE = 1.098075. It was 1.083289 before retraining.


################### Experimentation Part 3 #############

# I would choose these variables thinking they would positively impact on scoring higher grade: famsize+Pstatus+Medu+Fedu+Mjob+Fjob+studytime+activities+higher+internet

trainMyChoiceLM <- lm(G3~famsize+Pstatus+Medu+Fedu+Mjob+Fjob+studytime+activities+higher+internet, data = train.data)
summary(trainMyChoiceLM)

# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)   
# (Intercept)  -1.2522     3.2594  -0.384  0.70111   
# famsize       1.1462     0.5544   2.067  0.03953 * 
# Pstatus      -0.5269     0.8289  -0.636  0.52547   
# Medu          0.4333     0.3246   1.335  0.18293   
# Fedu          0.3877     0.2940   1.319  0.18815   
# Mjob         -0.1474     0.2411  -0.611  0.54145   
# Fjob          0.3909     0.3074   1.272  0.20439   
# studytime     0.5390     0.3004   1.794  0.07376 . 
# activities   -0.4531     0.5055  -0.896  0.37079   
# higher        3.1046     1.1533   2.692  0.00749 **
# internet      1.0004     0.6960   1.437  0.15162   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.402 on 305 degrees of freedom
# Multiple R-squared:  0.1041,	Adjusted R-squared:  0.07472 
# F-statistic: 3.544 on 10 and 305 DF,  p-value: 0.0001876

mseMyChoiceLMtrain<-mean(trainMyChoiceLM$residuals^2) # mean squared error (MSE)
mseMyChoiceLMtrain # 18.70653

# For test.data
testMyChoiceLM <- lm(G3~famsize+Pstatus+Medu+Fedu+Mjob+Fjob+studytime+activities+higher+internet, data = test.data)
summary(testMyChoiceLM)
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)   
# (Intercept)  -0.4484     7.3868  -0.061  0.95178   
# famsize       0.4577     1.2607   0.363  0.71767   
# Pstatus       1.1078     1.9613   0.565  0.57406   
# Medu          2.1301     0.6597   3.229  0.00191 **
# Fedu         -1.3664     0.6702  -2.039  0.04535 * 
# Mjob         -0.2230     0.4775  -0.467  0.64201   
# Fjob         -0.6017     0.6128  -0.982  0.32967   
# studytime    -0.3836     0.7006  -0.548  0.58578   
# activities    0.4381     1.0908   0.402  0.68921   
# higher        4.3264     2.7790   1.557  0.12416   
# internet     -0.3223     1.4318  -0.225  0.82261   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 4.477 on 68 degrees of freedom
# Multiple R-squared:  0.1698,	Adjusted R-squared:  0.04771 
# F-statistic: 1.391 on 10 and 68 DF,  p-value: 0.2034
mseMyChoiceLMtest<-mean(testMyChoiceLM$residuals^2) # mean squared error (MSE)
mseMyChoiceLMtest # 17.24992

# For logistic model with train.data
trainMyChoiceGLM <- glm(myGrade~famsize+Pstatus+Medu+Fedu+Mjob+Fjob+studytime+activities+higher+internet,family=binomial,data=xdata)
summary(trainMyChoiceGLM)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.6872  -1.1847   0.7522   1.0301   2.1623  
# 
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)   
# (Intercept) -4.79139    1.75886  -2.724  0.00645 **
# famsize      0.31356    0.26648   1.177  0.23933   
# Pstatus     -0.35479    0.40252  -0.881  0.37809   
# Medu        -0.02011    0.15308  -0.131  0.89546   
# Fedu         0.30662    0.14038   2.184  0.02894 * 
# Mjob         0.01671    0.11417   0.146  0.88362   
# Fjob         0.07694    0.14660   0.525  0.59970   
# studytime    0.22594    0.14309   1.579  0.11434   
# activities  -0.21742    0.24194  -0.899  0.36883   
# higher       1.44062    0.67794   2.125  0.03359 * 
# internet     0.64581    0.33457   1.930  0.05357 . 
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 437.04  on 315  degrees of freedom
# Residual deviance: 409.10  on 305  degrees of freedom
# AIC: 431.1
# 
# Number of Fisher Scoring iterations: 4
mseMyChoiceGLMtrain <- mean(resid(trainMyChoiceGLM)^2) # MSE
mseMyChoiceGLMtrain # 1.294616

# For logistic model with test.data
testMyChoiceGLM <- glm(myGrade1~famsize+Pstatus+Medu+Fedu+Mjob+Fjob+studytime+activities+higher+internet,family=binomial,data=xdata1)
summary(testMyChoiceGLM)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.8679  -1.2478   0.6742   0.9470   1.6248  
# 
# Coefficients:
#               Estimate Std. Error z value Pr(>|z|)
# (Intercept) -2.890293   3.644187  -0.793    0.428
# famsize      0.156015   0.612246   0.255    0.799
# Pstatus     -0.496345   0.999930  -0.496    0.620
# Medu         0.513621   0.335331   1.532    0.126
# Fedu        -0.472856   0.337948  -1.399    0.162
# Mjob        -0.005534   0.241546  -0.023    0.982
# Fjob         0.211968   0.293169   0.723    0.470
# studytime    0.181231   0.339067   0.534    0.593
# activities   0.306119   0.526680   0.581    0.561
# higher       1.549125   1.344546   1.152    0.249
# internet    -0.270676   0.714250  -0.379    0.705
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 102.723  on 78  degrees of freedom
# Residual deviance:  96.287  on 68  degrees of freedom
# AIC: 118.29
# 
# Number of Fisher Scoring iterations: 4
mseMyChoiceGLMtest <- mean(resid(testMyChoiceGLM)^2) # MSE
mseMyChoiceGLMtest # 1.218818

############### End ##############




