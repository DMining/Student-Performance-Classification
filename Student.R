setwd("C:/Users/Khoo Huai Swan/Desktop/Oct 2016/TDS3301 Data Mining/Assignment/Part3/Datasets")
matdf=read.table("student-mat.csv",sep=";",header=TRUE)
pordf=read.table("student-por.csv",sep=";",header=TRUE)

mergedf=merge(matdf,pordf,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(mergedf)) # 382 students

#Exploratory Data Analysis
#Math Course
#Number of students in each school
barplot(table(matdf$school), ylim = c(0, 350), main = "Number of Math Course Students in Each School", xlab = "School", ylab = "No. of Students")
barplot(table(pordf$school), ylim = c(0, 500), main = "Number of Portugese Course Students in Each School", xlab = "School", ylab = "No. of Students")
#Age distribution
barplot(table(matdf$age), ylim = c(0,120), main= "Age Distribution (Math Course)", ylab = "Age",xlab = "Number of students")
barplot(table(pordf$age), ylim = c(0,200), main= "Age Distribution (Portugese Course)", ylab = "Age",xlab = "Number of students")

#Grade 
#Average Grade for each student
rowMeans(matdf[,31:33])
rowMeans(pordf[,31:33])
#Average grade for all students
aveGradeAll_mat <- sum(matdf$aveGrade)/nrow(matdf)
aveGradeAll_por <- sum(pordf$aveGrade)/nrow(pordf)

#Pre-processing Tasks
matdf$failpass <- factor(with(matdf, ifelse(G3<10, "Fail", "Pass")))
pordf$failpass <- factor(with(pordf, ifelse(G3<10, "Fail", "Pass")))

str(matdf)
str(pordf)

#Partition into Train and Test sets
size_mat <- floor(0.7 * nrow(matdf))
size_por <- floor(0.7 * nrow(pordf))

#set seed
set.seed(10)
train_ind_mat <- sample(seq_len(nrow(matdf)), size = size_mat)
train_ind_por <- sample(seq_len(nrow(pordf)), size = size_por)

train_mat <- matdf[train_ind_mat, ]
test_mat <- matdf[-train_ind_mat, ]

train_por <- pordf[train_ind_por, ]
test_por <- pordf[-train_ind_por, ]

########################################Classification ################################
################Decision Tree
install.packages("rpart")
install.packages("rpart.plot")
install.packages("rattle")
library(rpart)
library(rpart.plot)
library(rattle)
################### Math course ####################
#Train the tree
model_mat <- rpart(failpass~ . -G3, data=train_mat)
#Plot the tree
plot(model_mat)
text(model_mat,pretty=0, cex=0.7)

#Better plot using rpart.plot(prp) and rattle(fancyRPlot)
prp(model_mat)
fancyRpartPlot(model_mat)
predictrpart <- predict(model_mat, test_mat, type="class")
plot(predictrpart, ylim = c(0,100))
mean(predictrpart == test_mat$failpass)

#Examine cross validation result
printcp(model_mat)
plotcp(model_mat)

#Pruning data
prunedmodel_mat <- prune(model_mat, cp= model_mat$cptable[which.min(model_mat$cptable[,"xerror"]),"CP"])
predictrpruned_mat <- predict(prunedmodel_mat, test_mat, type="class")
plot(predictrpruned_mat)
text(prunedmodel_mat,pretty=0, cex=0.7)
plotcp(model_mat)
prp(prunedmodel_mat)

###############Portugese course###############
#Train tree
model_por <- rpart(failpass~ . -G3, data=train_por)
#Plot the tree
plot(model_por)
text(model_por,pretty=0, cex=0.7)

#Better plot using rpart.plot(prp) and rattle(fancyRPlot)
prp(model_por)
fancyRpartPlot(model_por)
predictrpart_por <- predict(model_por, test_por, type="class")
plot(predictrpart_por, ylim = c(0,200))
mean(predictrpart_por == test_por$failpass)

#Examine cross validation result
printcp(model_por)
plotcp(model_por)

#Pruning data
prunedmodel_por <- prune(model_por, cp= model_por$cptable[which.min(model_por$cptable[,"xerror"]),"CP"])
predictrpruned_por <- predict(prunedmodel_por, test_por, type="class")
plot(predictrpruned_por)
text(prunedmodel_por,pretty=0, cex=0.7)
plotcp(model_por)
prp(prunedmodel_por)

# Random Forest
library(randomForest)
rfModel_mat <- randomForest(failpass~ . -G3, data=train_mat)
print(rfModel_mat)  # view results 
importance(rfModel_mat) 


#######################Naive Bayes################################
##matdf##############################
#check number of pass and fail
table(matdf$failpass)

library(caret)
set.seed(500) #set seed to get random sample

#partition dataset into training and testing sets
#70% for training
matdf.size<-length(matdf$failpass)
matdf.train.size<-round(matdf.size*0.7)
matdf.testing.size<-matdf.size-matdf.train.size
matdf.train.idx<-sample(seq(1:matdf.size), matdf.train.size)
matdf.train.sample<-matdf[matdf.train.idx,]
matdf.testing.sample<-matdf[-matdf.train.idx,]

#check number of rows to see if have randomly selected training and testing
nrow(matdf.train.sample)
nrow(matdf.testing.sample)

#number of pass and fail in training set
table(matdf.train.sample$failpass)

library(e1071)
library(rminer)

#modelbuilding
matdf.model<-naiveBayes(failpass ~ ., data=matdf.train.sample)
#check summary of model
#for each factor variable, table of likelihood is calculated
#for numeric variable, average number is calculated
matdf.model

#conditional probability of school
prop.table(table(matdf.train.sample$school, matdf.train.sample$G3))

#frequency of school
table(matdf.train.sample$school, matdf.train.sample$failpass)

#prediction on testing set
matdf.prediction<-predict(matdf.model, matdf.testing.sample)

head(matdf.prediction)

#print result of confusion matrix
#accuracy and sensitivity are good
print(confusionMatrix(matdf.prediction, matdf.testing.sample$failpass, positive="Pass", dnn=c("Prediction", "True")))


#################################################################pordf##########################

table(pordf$failpass)

set.seed(500)

#partition dataset into training and testing sets
#70% for training
pordf.size<-length(pordf$failpass)
pordf.train.size<-round(pordf.size*0.7)
pordf.testing.size<-pordf.size - pordf.train.size
pordf.train.idx<-sample(seq(1:pordf.size), pordf.train.size)
pordf.train.sample<-pordf[pordf.train.idx,]
pordf.testing.sample<-pordf[-pordf.train.idx,]

nrow(pordf.train.sample)
nrow(pordf.testing.sample)

table(pordf.train.sample$failpass)

pordf.model<-naiveBayes(failpass ~ ., data=pordf.train.sample)
pordf.model

prop.table(table(pordf.train.sample$school, pordf.train.sample$G3))

table(pordf.train.sample$school, pordf.train.sample$failpass)

pordf.prediction<-predict(pordf.model, pordf.testing.sample)

head(pordf.prediction)

#accuracy and sensitivity are good
print(confusionMatrix(pordf.prediction, pordf.testing.sample$failpass, positive="Pass", dnn=c("Prediction", "True")))


#################ANN#################################################################################

library(ISLR)
head(matdf)

#Section 1 exploratory data analysis
dataset<-matdf
str(dataset)
is.na(dataset)
##head(dataset[c(1:10), c(6:7,11)])#rows 1 to 10, columns 6-7 and 11
##tail(dataset[c(1:10), c(6:7,11)])#rows 1 to 10, columns 6-7 and 11
any(is.na(dataset))# this will stop after the first NA instead of going through the entire vector as would be the case with any(is.na())
boxplot(dataset)#boxplot
#any test are acceptable , skew etc

#Section 2 scaling entire dataframe
# Create Vector of Column Max and Min Values
G3=as.data.frame(lapply(dataset[33], normalize))

normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x))) }
scaled.data <- as.data.frame(lapply(dataset[31:32], normalize))
data<-cbind(scaled.data, G3) 

##maxs <- apply(dataset[,2:18], 2, max)
##mins <- apply(dataset[,2:18], 2, min)
# Use scale() and convert the resulting matrix to a data frame
##scaled.data <- as.data.frame(scale(dataset[,2:18],center = mins, scale = maxs - mins))

#Section 3 : Training and testing data
#Convert Private column from Yes/No to 1/0
##Private = as.numeric($Private)-1
##data = cbind(Private,scaled.data)
library(caTools)
set.seed(101)
# Create Split (any column is fine)
split = sample.split(data$G3, SplitRatio = 0.70)
# Split based off of split Boolean Vector
train = subset(data, split == TRUE)
test = subset(data, split == FALSE)

#Section 4: 
feats <- names(scaled.data)
# Concatenate strings
f <- paste(feats,collapse=' + ')
f <- paste('G3 ~',f)
# Convert to formula
f <- as.formula(f)
f

#Section 5: Create NN model
#install.packages('neuralnet')
library(neuralnet)
nn <- neuralnet(f,data,hidden=4,linear.output=FALSE)
# Compute Predictions off Test Set
predicted.nn.values <- compute(nn,test[1:2])
# Check out net.result
print(head(predicted.nn.values$net.result))
predicted.nn.values$net.result <- sapply(predicted.nn.values$net.result,round,digits=0)
table(test$G3,predicted.nn.values$net.result)
plot(nn)

