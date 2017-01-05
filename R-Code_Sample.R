#############################################################
#Question 1
#############################################################
#(a)
oj = read.csv("OrangeJuice.csv")
attach(oj)

set.seed(4754)
tmp=sample(1:nrow(oj),0.75*nrow(oj))
#25% test
test = -tmp

#number of elements in our training set
num.train = floor((2/3)*length(tmp))
#training set
train = tmp[1:num.train] #make train first num.train elements
#validation set
valid = tmp[(num.train+1):length(tmp)]

#summary of training data
summary(oj[train,])

#(b)
#Logistic Regression
logit.fit = glm(Purchase ~. -X, data = oj[train,], family = "binomial")
summary(logit.fit)

#(c)
#First, we remove linearly dependent columns we don't think will be useful
logit.fit = glm(Purchase ~. -X -PriceCH - PriceMM - DiscCH - DiscMM - SalePriceMM - SalePriceCH, data = oj[train,], family = "binomial")
summary(logit.fit)
#Then, we use backward stepwise selection to find the two variables with the most
#predictive power, iteratively eliminating the variable with the highest p-value
#Removing StoreID
logit.fit = glm(Purchase ~. -X -PriceCH - PriceMM - DiscCH - DiscMM - SalePriceMM - SalePriceCH -StoreID, data = oj[train,], family = "binomial")
summary(logit.fit)
#Removing WeekofPurchase
logit.fit = glm(Purchase ~. -X -PriceCH - PriceMM - DiscCH - DiscMM - SalePriceMM - SalePriceCH -StoreID -WeekofPurchase, data = oj[train,], family = "binomial")
summary(logit.fit)
#etc.
#Final Model:
logit.fit = glm(Purchase ~ LoyalCH + PriceDiff, data = oj[train,], family = "binomial")
summary(logit.fit)

#(d)
library(tree)
tree.fit = tree(Purchase ~. -X, data = oj[train,])
#plot the full tree
plot(tree.fit)
text(tree.fit,pretty=0)
# we prune the tree to prevent overfitting
cv.fit = cv.tree(tree.fit, FUN=prune.misclass)
plot(cv.fit$size,cv.fit$dev,type="b")
min.index = which(cv.fit$dev == min(cv.fit$dev))
opt.num.terminal.nodes = cv.fit$size[min.index] #the optimal number of terminal nodes
#in case if the optimal isn't unique, choose the smaller one
opt.num.terminal.nodes = min(opt.num.terminal.nodes)
#create a pruned tree with opt.num.terminal.nodes
prune.fit = prune.misclass(tree.fit,best=opt.num.terminal.nodes)
#plot the pruned tree
plot(prune.fit)
text(prune.fit,pretty=0)

#(e)
library(e1071)
C.vec = c(0.01, 0.1, 1,10,100)
#prediction accuracies:
num.correct.vec = rep(0,length(C.vec))
acc.vec = rep(0,length(C.vec))
for (i in 1:length(C.vec)) {
  C = C.vec[i]
  svm.fit=svm(Purchase ~. -X, data=oj[train,], kernel="linear", cost=C)
  svm.decision=predict(svm.fit,oj[valid,])
  svm.table = table(truth=oj[valid,"Purchase"], predict=svm.decision)
  num.correct.vec[i] = svm.table[1,1] + svm.table[2,2]
  acc.vec[i] = (svm.table[1,1] + svm.table[2,2])/sum(svm.table)
}
#choosing model with max prediction accuracy on validation set
best.i = which.max(acc.vec)
svm.best.acc = acc.vec[best.i]
svm.best.C = C.vec[best.i]

#(f)
#Logistic Regression assessment
logit.pred.prob = predict(logit.fit,newdata = oj[valid,], type = "response")

#Transform the predicted probability to a decision and test it
p = 0.50;
logit.decision = ifelse(logit.pred.prob > p,'MM','CH')
logit.table = table(truth=oj[valid,"Purchase"], predict=logit.decision)
#logit.table
logit.acc = (logit.table[1,1] + logit.table[2,2])/sum(logit.table)
logit.acc

#decision tree assessment
tree.pred=predict(prune.fit, oj[valid,],type="class")
tree.table=table(tree.pred,oj[valid,"Purchase"])
tree.acc = (tree.table[1,1] + tree.table[2,2])/sum(tree.table)
tree.acc

#SVM assessment:
svm.best.acc

#(g)
#logistic regression
#refit model on the training+validation sets
logit.fit = glm(Purchase ~ LoyalCH + PriceDiff, data = oj[c(train,valid),], family = "binomial")
logit.pred.prob = predict(logit.fit,newdata = oj[test,], type = "response")
p = 0.5 #decision threshold
logit.decision = ifelse(logit.pred.prob > p,'MM','CH')
logit.table = table(truth=oj[test,"Purchase"], predict=logit.decision)
logit.test.acc = (logit.table[1,1] + logit.table[2,2])/sum(logit.table)
logit.test.acc

#decision trees
tree.fit = tree(Purchase ~. -X, data = oj[c(train,valid),])
prune.fit = prune.misclass(tree.fit,best=opt.num.terminal.nodes)
tree.pred=predict(prune.fit, oj[test,],type="class")
tree.table=table(tree.pred,oj[test,"Purchase"])
tree.test.acc = (tree.table[1,1] + tree.table[2,2])/sum(tree.table)
tree.test.acc

#SVM
#Fitting SVM on the training+validation data
svm.fit=svm(Purchase ~. -X, data=oj[c(train,valid),], kernel="linear", cost=svm.best.C)
svm.decision=predict(svm.fit,oj[test,])
svm.table = table(truth=oj[test,"Purchase"], predict=svm.decision)
svm.test.acc = (svm.table[1,1] + svm.table[2,2])/sum(svm.table) 
svm.test.acc

#(h)
#We want to offer coupons to customers who we predict drank CH
#The profit matrix for this problem is as follows:
#decision = coupon, truth = CH: $3
#decision = no coupon, truth = CH: $0
#decision = coupon, truth = MM: $-1
#decision = no coupon, truth = MM: $0
profit.matrix=matrix(c(3,0,-1,0),nrow=2,ncol=2,byrow=TRUE)

#logistic regression: finding the optimal threshold p
#sequence of decision thresholds to test:
possible.ps = seq(from=0.05,to=0.95,by=0.01)
#logit.profits[i] corresponds to the profit from using threshold p=possible.ps[i]
logit.profits = rep(0, times=length(possible.ps))
for (i in 1:length(possible.ps)) {
  p = possible.ps[i]
  logit.decision = ifelse(logit.pred.prob > p,'MM','CH')
  logit.table = table(truth=oj[test,"Purchase"], predict=logit.decision)
  logit.profits[i] = sum(logit.table * profit.matrix)
}
plot(possible.ps,logit.profits)
#best decision threshold
logit.best.p = possible.ps[which.max(logit.profits)]
logit.best.p
#maximum profit corresponding to best.p
logit.best.profit = max(logit.profits)
logit.best.profit

#decision trees: finding the optimal threshold p
tree.pred.prob=predict(prune.fit, oj[test,],type="vector")[,"MM"]
#sequence of decision thresholds to test:
possible.ps = seq(from=0.10,to=0.85,by=0.01)
#tree.profits[i] corresponds to the profit from using threshold p=possible.ps[i]
tree.profits = rep(0, times=length(possible.ps))
for (i in 1:length(possible.ps)) {
  p = possible.ps[i]
  tree.decision = ifelse(tree.pred.prob > p,'MM','CH')
  tree.table = table(truth=oj[test,"Purchase"], predict=tree.decision)
  tree.profits[i] = sum(tree.table * profit.matrix)
}
plot(possible.ps,tree.profits)
#best decision threshold
tree.best.p = possible.ps[which.max(tree.profits)]
tree.best.p
#maximum profit corresponding to best.p
tree.best.profit = max(tree.profits)
tree.best.profit

#SVM
#Fitting SVM on the training+validation data
svm.fit=svm(Purchase ~. -X, data=oj[c(train,valid),], kernel="linear", cost=svm.best.C)
svm.decision=predict(svm.fit,oj[test,])
svm.table = table(truth=oj[test,"Purchase"], predict=svm.decision)
svm.best.profit = sum(svm.table * profit.matrix) 
svm.best.profit

detach(oj)
#############################################################
#Question 2
#############################################################
set.seed(4574)
#(a)
#original dataset "cm"
cm = read.csv("classifyMe.csv")
summary(cm)
####################################################################
#Run this code to obtain original, unmodified outier
#cm[nrow(cm),"X1"]=cm[nrow(cm),"X1"]+1000
#cm[nrow(cm),"X2"]=cm[nrow(cm),"X2"]+1000
####################################################################
#removing the outlier (call this dataset cm2)
cm2 = cm[1:(nrow(cm)-1),]
summary(cm2)

#(b)
#removing "X" column of cm and cm2
cm = cm[,-1]
cm2 = cm2[,-1]

library('MASS')
#LDA on cm
lda.fit = lda(y ~., data = cm)
lda.fit

#LDA on cm2
lda.fit2 = lda(y ~., data = cm2)
lda.fit2

#(c)
library(e1071)
#SVM on cm
tune.out=tune(svm,y ~.,data=cm,kernel="linear",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,10,100)),
              tunecontrol=tune.control(sampling=c("cross"),cross=10))
svm.fit=tune.out$best.model
summary(svm.fit)
#plot(svm.fit,cm)
plot(svm.fit,cm[-nrow(cm),])
#SVM on cm2
tune.out=tune(svm,y ~.,data=cm2,kernel="linear",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,10,100)),
              tunecontrol=tune.control(sampling=c("cross"),cross=10))
svm.fit2=tune.out$best.model
summary(svm.fit2)
plot(svm.fit2,cm2)

#(d)
cm.test = read.csv("classifyMeValidation.csv")
#removing "X" column of cm.test
cm.test = cm.test[,-1]

#LDA on cm
#Obtain the predicted probabilities
lda.pred.prob = predict(lda.fit,newdata = cm.test)
lda.pred.prob = lda.pred.prob$posterior[,"Y"]
#Transform the predicted probability to a decision and test it
p = 0.50;
lda.decision = ifelse(lda.pred.prob > p,"Y","N")
lda.table = table(truth=cm.test[,"y"], predict=lda.decision)
lda.num.correct = lda.table[1,1] + lda.table[2,2]
lda.num.correct
lda.acc = (lda.table[1,1] + lda.table[2,2])/sum(lda.table)
lda.acc

#LDA on cm2
#Obtain the predicted probabilities
lda.pred.prob2 = predict(lda.fit2,newdata = cm.test)
lda.pred.prob2 = lda.pred.prob2$posterior[,"Y"]
#Transform the predicted probability to a decision and test it
p = 0.50;
lda.decision2 = ifelse(lda.pred.prob2 > p,"Y","N")
lda.table2 = table(truth=cm.test[,"y"], predict=lda.decision2)
lda.num.correct2 = lda.table2[1,1] + lda.table2[2,2]
lda.num.correct2
lda.acc2 = (lda.table2[1,1] + lda.table2[2,2])/sum(lda.table2)
lda.acc2

#SVM on cm
svm.decision=predict(svm.fit,cm.test)
svm.table = table(truth=cm.test[,"y"], predict=svm.decision)
svm.num.correct = svm.table[1,1] + svm.table[2,2]
svm.num.correct
svm.acc = (svm.table[1,1] + svm.table[2,2])/sum(svm.table) 
svm.acc

#SVM on cm2
svm.decision=predict(svm.fit2,cm.test)
svm.table = table(truth=cm.test[,"y"], predict=svm.decision)
svm.num.correct2 = svm.table[1,1] + svm.table[2,2]
svm.num.correct2
svm.acc2 = (svm.table[1,1] + svm.table[2,2])/sum(svm.table) 
svm.acc2
#############################################################
#Question 3
#############################################################
#(a)
cp = read.csv("cuisinePreferences.csv")
# first column is the names, make it the row labels instead
rownames(cp) = cp[,1]
# remove first row
cp = cp[,-1]
head(cp)
################################
#parameter specifying the section of the student
mySection = 2
#adjust this parameter accordingly
filename = "Section 2 - Prediction Matrix.csv"
################################
#create the training data using students in this section
cp.train = cp[cp$Section == mySection, ]
#create the test data using students in the other section
cp.test = cp[cp$Section != mySection, ]
#delete the section column from both datasets
cp.train = cp.train[,-ncol(cp.train)]
cp.test = cp.test[,-ncol(cp.test)]

#convert to matrix format
#(transforming to a matrix helps speed up some of the calculations below)
cp.train = as.matrix(cp.train)
cp.test = as.matrix(cp.test)

#(b)
# n = number of students
n = nrow(cp.train)
# m = number of cuisines
m = ncol(cp.train)


# Calculate similarity between each two students

sim = matrix(NA,nrow = n, ncol = n) 
for (i in 1:n)
{
  for (j in 1:n)
  {
    d = cp.train[i,] - cp.train[j,]
    sim[i,j] = sqrt(sum(d*d, na.rm = TRUE))
  }
}
colnames(sim) = rownames(cp.train)
rownames(sim) = rownames(cp.train)

# Find 5 closest students to each student

for (u in 1:nrow(sim)) {
  # auxiliary function, used to generate the tie breaking rule
  listUfirst<- function(u,n){
    if (u>1 & u < n)
      return( c(u,1:(u-1),(u+1):n))
    else
      if (u==1)
        return(1:n)
    else
      return(c(n,1:(n-1)))
  }

  # for each user, create a table of rankings where the top rows are the most similar users
  orderedCP = cp.train[order(sim[u,],listUfirst(u,n)),]
  # print the closest 5 students
  closest.five.students = rownames(orderedCP[2:6,])
  print(closest.five.students)
}

#print order of students we looped over
for (u in 1:nrow(sim)) {
  print(rownames(sim)[u])
}

#(c)

# This function uses the nearest neighbor method to predict a users ranking of a cuisine
# It uses common ratings to define distance between users, 
# and returns the average ranking of the closest nn users who ranked the cuisine
## Parameters:
# nn - number of nearest neighbors to use
# cp.train - matrix of ratings data. Unrated cuisines should be recorded as NA.

predictRatings<- function(nn, cp.train){
  
  # n = number of users
  n = nrow(cp.train)
  # m = number of movies
  m = ncol(cp.train)
  
  sim = matrix(NA,nrow = n, ncol = n) 
  for (i in 1:n)
  {
    for (j in 1:n)
    {
      d = cp.train[i,] - cp.train[j,]
      sim[i,j] = sqrt(sum(d*d, na.rm = TRUE))
    }
  }
  
  predictionMatrix = data.frame()
  for (u in 1:n)
  {
    
    tieBreak = c(u,1:n)[!duplicated(c(u,1:n))]
    orderedMovieRanks = cp.train[order(sim[u,],tieBreak),]
    
    for (mov in 1:m)
    {
      ratings = na.omit(orderedMovieRanks[2:n,mov])
      ratings = as.vector(ratings)
      if (length(ratings) > nn )
        predictionMatrix[u,mov] = mean(ratings[1:nn])
      else
        predictionMatrix[u,mov] = mean(ratings)
      
    }
  }
  colnames(predictionMatrix) = colnames(cp.train)
  row.names(predictionMatrix) = row.names(cp.train)
  
  return(predictionMatrix) 
}

# Produce a prediction and format the row and column labels
predictionMatrix = predictRatings(3, cp.train)

# Write to a .csv
tmp = predictionMatrix
rownames(tmp) = NULL
colnames(tmp) = NULL
tmp = matrix(as.matrix(tmp), ncol = ncol(tmp), dimnames = NULL)
write.table(tmp,filename,row.names = FALSE, col.names = FALSE, sep=",")

#(d)
# Test for best nn value

# this vector holds the RMSE for each value of nn in 1:20
RMSEperformance = vector(length = 20)

# calculate the RSME for each value of nn
for (nn in 1:20){
  predictionMatrix = predictRatings(nn, cp.train)
  RMSEperformance[nn] = sqrt(mean((predictionMatrix - cp.train)^2,na.rm = TRUE))
}

# plot the result
plot(1:20, RMSEperformance,xlab="Number of Neighbors")

#best result
nn.best = which.min(RMSEperformance)

#(e)
#obtain prediction for all students from the other section using our model
#create similarity matrix for all students in other section
sim = matrix(NA,nrow = nrow(cp.test), ncol = n)
for (i in 1:nrow(cp.test))
{
  for (j in 1:n)
  {
    d = cp.test[i,] - cp.train[j,]
    sim[i,j] = sqrt(sum(d*d, na.rm = TRUE))
  }
}
colnames(sim) = rownames(cp.train)
rownames(sim) = rownames(cp.test)
#obtain predictions
m = ncol(cp.train)
predictionMatrix = data.frame()
for (u in 1:nrow(cp.test))
{
  orderedMovieRanks = cp.train[order(sim[u,]),]
  
  for (mov in 1:m)
  {
    ratings = na.omit(orderedMovieRanks[2:n,mov])
    ratings = as.vector(ratings)
    if (length(ratings) > nn.best )
      predictionMatrix[u,mov] = mean(ratings[1:nn.best])
    else
      predictionMatrix[u,mov] = mean(ratings)
    
  }
}
colnames(predictionMatrix) = colnames(cp.test)
row.names(predictionMatrix) = row.names(cp.test)

#########################################
#parameter specifying identities of 3 students
u.vec = c(1,2,3)
#########################################

RMSEperformance = sqrt(mean((predictionMatrix[u.vec,] - cp.test[u.vec,])^2,na.rm = TRUE))



