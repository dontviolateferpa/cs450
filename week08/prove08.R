library(e1071)
library(MASS)
library(readr)
library(caret)

# Initialize letter
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
print(sum(diag(tab)))

# Run #2
model2 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.001, cost = 10)
summary(model2)
prediction2 <- predict(model2, testset_letters[,-1])
tab2 <- table(pred = prediction2, true = testset_letters[,1])
print(tab2)
print(sum(diag(tab2)))

# Run #3
model3 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.01, cost = 5)
summary(model3)
prediction3 <- predict(model3, testset_letters[,-1])
tab3 <- table(pred = prediction3, true = testset_letters[,1])
print(tab3)
print(sum(diag(tab3)))


# Run #4
model4 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.03, cost = 2)
summary(model4)
prediction4 <- predict(model4, testset_letters[,-1])
tab4 <- table(pred = prediction4, true = testset_letters[,1])
print(tab4)
print(sum(diag(tab4)))

# Run #5
model5 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.02, cost = 4)
summary(model5)
prediction5 <- predict(model5, testset_letters[,-1])
tab5 <- table(pred = prediction5, true = testset_letters[,1])
print(tab5)
print(sum(diag(tab5)))

# Run #6
model6 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.5, cost = 5)
summary(model6)
prediction6 <- predict(model6, testset_letters[,-1])
tab6 <- table(pred = prediction6, true = testset_letters[,1])
print(tab6)
print(sum(diag(tab6)))

# Run #7
model7 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.017, cost = 22)
summary(model7)
prediction7 <- predict(model7, testset_letters[,-1])
tab7 <- table(pred = prediction7, true = testset_letters[,1])
print(tab7)
print(sum(diag(tab7)))

# Run #8
model8 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.022, cost = 13)
summary(model8)
prediction8 <- predict(model8, testset_letters[,-1])
tab8 <- table(pred = prediction8, true = testset_letters[,1])
print(tab8)
print(sum(diag(tab8)))

# Run #9
model9 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.07, cost = 5)
summary(model9)
prediction9 <- predict(model9, testset_letters[,-1])
tab9 <- table(pred = prediction9, true = testset_letters[,1])
print(tab9)
print(sum(diag(tab9)))

# Run #10
model10 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.09, cost = 30)
summary(model10)
prediction10 <- predict(model10, testset_letters[,-1])
tab10 <- table(pred = prediction10, true = testset_letters[,1])
print(tab10)
print(sum(diag(tab10)))

