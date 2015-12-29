# titanicExploration.R
# This script is for exploring the data sets associated
# with the kaggle.com Titanic competition.

library(ggplot2) # for plotting
library(class) # for k nearest neighbors clustering
library(caret) # for data slicing
# load in the training set
train <- read.csv("train.csv", stringsAsFactors = FALSE)

# do some basic data transformations
train$Sex <- as.factor(train$Sex)
train$Embarked <- as.factor(train$Embarked)
train$Pclass <- as.factor(train$Pclass)
train$Survived <- as.factor(train$Survived)

# do some exploration
str(train)
summary(train)
hist(train$Age, breaks = 20)
hist(train$Fare, breaks = 20)
hist(train$SibSp)
hist(train$Parch)
hist(train$Embarked)
qplot(Age, Survived, data = train) # not an obvious relationship
qplot(Age, Survived, data = train, colour = Sex) # many more women survived
qplot(Age, Fare, data = train, colour = Survived)
qplot(Age, Sex, data = train, colour = Survived) # appears lower fares may have been less likely to survive
qplot(Fare, Pclass, data = train, colour = Survived) # appears to be more survivors in higher classes
qplot(Age, Pclass, data = train, colour = Survived)
qplot(Sex, Pclass, data = train, colour = Survived)
qplot(SibSp, Parch, data = train, colour = Survived)

# look for correlations between numeric columns using findCorrelation()
# first, exclude character columns from the set
nums <- sapply(train, is.numeric)
trainnums <- train[, nums]
traincor <- findCorrelation(cor(trainnums[,1:length(trainnums)-1]), cutof = .8, verbose = FALSE)
traincor # apparently no obvious correlation between numeric columns
#rm(trainnums)
#rm(nums)

# prepping data for knn and predictive analysis
set.seed(333)

# rescaling (normalize) the data so variables with wide ranges do not have undue influence 
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

trainNorm <- train

# address NAs in age column
meanAge <- mean(trainNorm[!is.na(trainNorm$Age),6])
trainNorm[is.na(trainNorm$Age),6] <- meanAge

# normalize all numeric, independent variables
trainNorm$Age <- normalize(trainNorm$Age)
trainNorm$Fare <- normalize(trainNorm$Fare)

# use random sampling to split the data set
ind <- sample(2, nrow(trainNorm), replace = TRUE, prob = c(0.67, 0.33))
train2 <- trainNorm[ind == 1,]
test2 <- trainNorm[ind==2,]

# do some KNN
#sqrt(nrow(trainNorm)) # find k
trainknn <- train2[, c(2,6:8,10)]
testknn <- test2[, c(2,6:8,10)]
m1 <- knn(train = trainknn[,2:5], test = testknn[,2:5], cl = trainknn[,1], k = 29)
confusionMatrix(m1, testknn[,1]) # only .70 accuracy

# try some linear models
# use step models/AIC to analyze impact of different features
# may need to change independent factor variables to numeric or dummy variables

# try gbm

# try random forest

# run an ensemble model and submit
