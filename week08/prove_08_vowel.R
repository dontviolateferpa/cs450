library(e1071)
library(MASS)
library(readr)
library(caret)

# Initialize vowel
datavowel = read.csv("C:\\Users\\Zachary\\OneDrive\\Documents\\09 BYUI 2018 Winter (Override)\\CS 450 - Machine Learning and Data Mining\\cs450\\week08\\vowel.csv")
index <- 1:nrow(datavowel)
testindex_vowel <- sample(index, trunc(length(index)/3))
testset_vowel <- datavowel[testindex_vowel,]
trainset_vowel <- datavowel[-testindex_vowel,]

# Run #1
model_vowel <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.0625, cost = 1)
summary(model_vowel)
prediction <- predict(model_vowel, testset_vowel[,-1])
tab <- table(pred = prediction, true = testset_vowel[,1])
print(tab)
print(sum(diag(tab)))

# Run #2
model_vowel2 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.001, cost = 10)
summary(model_vowel2)
prediction2 <- predict(model_vowel2, testset_vowel[,-1])
tab2 <- table(pred = prediction2, true = testset_vowel[,1])
print(tab2)
print(sum(diag(tab2)))

# Run #3
model_vowel3 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.01, cost = 5)
summary(model_vowel3)
prediction3 <- predict(model_vowel3, testset_vowel[,-1])
tab3 <- table(pred = prediction3, true = testset_vowel[,1])
print(tab3)
print(sum(diag(tab3)))


# Run #4
model_vowel4 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.03, cost = 2)
summary(model_vowel4)
prediction4 <- predict(model_vowel4, testset_vowel[,-1])
tab4 <- table(pred = prediction4, true = testset_vowel[,1])
print(tab4)
print(sum(diag(tab4)))

# Run #5
model_vowel5 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.02, cost = 4)
summary(model_vowel5)
prediction5 <- predict(model_vowel5, testset_vowel[,-1])
tab5 <- table(pred = prediction5, true = testset_vowel[,1])
print(tab5)
print(sum(diag(tab5)))

# Run #6
model_vowel6 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.5, cost = 5)
summary(model_vowel6)
prediction6 <- predict(model_vowel6, testset_vowel[,-1])
tab6 <- table(pred = prediction6, true = testset_vowel[,1])
print(tab6)
print(sum(diag(tab6)))

# Run #7
model_vowel7 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.017, cost = 22)
summary(model_vowel7)
prediction7 <- predict(model_vowel7, testset_vowel[,-1])
tab7 <- table(pred = prediction7, true = testset_vowel[,1])
print(tab7)
print(sum(diag(tab7)))

# Run #8
model_vowel8 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.022, cost = 13)
summary(model_vowel8)
prediction8 <- predict(model_vowel8, testset_vowel[,-1])
tab8 <- table(pred = prediction8, true = testset_vowel[,1])
print(tab8)
print(sum(diag(tab8)))

# Run #9
model_vowel9 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.07, cost = 5)
summary(model_vowel9)
prediction9 <- predict(model_vowel9, testset_vowel[,-1])
tab9 <- table(pred = prediction9, true = testset_vowel[,1])
print(tab9)
print(sum(diag(tab9)))

# Run #10
model_vowel10 <- svm(Class~., data = datavowel, kernel = "radial", gamma = 0.09, cost = 30)
summary(model_vowel10)
prediction10 <- predict(model_vowel10, testset_vowel[,-1])
tab10 <- table(pred = prediction10, true = testset_vowel[,1])
print(tab10)
print(sum(diag(tab10)))
