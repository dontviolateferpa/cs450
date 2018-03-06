library(e1071)
library(MASS)
library(readr)
library(caret)

# Initialize
dataletters = read.csv("C:\\Users\\Zachary\\OneDrive\\Documents\\09 BYUI 2018 Winter (Override)\\CS 450 - Machine Learning and Data Mining\\cs450\\week08\\letters.csv")
index <- 1:nrow(dataletters)
testindex_letters <- sample(index, trunc(length(index)/3))
testset_letters <- dataletters[testindex_letters,]
trainset_letters <- dataletters[-testindex_letters,]

# Run #1
model <- svm(letter~., data = trainset_letters)
summary(model)
prediction <- predict(model, testset_letters[,-1])
tab <- table(pred = prediction, true = testset_letters[,1])
print(tab)

# Run #2
model2 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.001, cost = 10)
summary(model2)
prediction2 <- predict(model2, testset_letters[,-1])
tab2 <- table(pred = prediction2, true = testset_letters[,1])
print(tab2)
print(sum(diag(tab)))
