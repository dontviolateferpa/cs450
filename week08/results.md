# datasets and results

## letters

### combination #1

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  1 
      gamma:  0.0625 

Number of Support Vectors:  9987
```

#### accuracy

86.9%

### combination #2

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  10 
      gamma:  0.001 

Number of Support Vectors:  13915
```

#### accuracy

83.9%

### combination #3

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  5 
      gamma:  0.01 

Number of Support Vectors:  10169
```

#### accuracy

92.4%

### combination #4

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  2 
      gamma:  0.03 

Number of Support Vectors:  9683
```

#### accuracy

94.96%

### combination #5

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  4 
      gamma:  0.02 

Number of Support Vectors:  9138
```

#### accuracy

94.8%

### combination #6

#### parameters

```txt
# Run #5
model5 <- svm(letter~., data = dataletters, kernel = "radial", gamma = 0.02, cost = 4)
summary(model5)
prediction5 <- predict(model5, testset_letters[,-1])
tab5 <- table(pred = prediction5, true = testset_letters[,1])
print(tab5)
print(sum(diag(tab5)))
```

#### accuracy

100%

### combination #7

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  22 
      gamma:  0.017 

Number of Support Vectors:  7034
```

#### accuracy

97.4%

### combination #8

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  13 
      gamma:  0.022 

Number of Support Vectors:  7314
```

#### accuracy

97.5%

### combination #9

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  5 
      gamma:  0.07 

Number of Support Vectors:  7940
```

#### accuracy

98.9%

### combination #10

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  30 
      gamma:  0.09 

Number of Support Vectors:  7897
```

#### accuracy

99.9%

## vowel

### combination #1

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  1 
      gamma:  0.0625 

Number of Support Vectors:  848
```

#### accuracy

98.1%

### combination #2

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  10 
      gamma:  0.001 

Number of Support Vectors:  959
```

#### accuracy

74.2%

### combination #3

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  5 
      gamma:  0.01 

Number of Support Vectors:  838
```

#### accuracy

94.5%

### combination #4

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  2 
      gamma:  0.03 

Number of Support Vectors:  823
```

#### accuracy

97%

### combination #5

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  4 
      gamma:  0.02 

Number of Support Vectors:  764
```

#### accuracy

97.6%

### combination #6

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  5 
      gamma:  0.5 

Number of Support Vectors:  725
```

#### accuracy

100%

### combination #7

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  22 
      gamma:  0.017 

Number of Support Vectors:  558
```

#### accuracy

99.1%

### combination #8

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  13 
      gamma:  0.022 

Number of Support Vectors:  600
```

#### accuracy

99.1%

### combination #9

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  5 
      gamma:  0.07 

Number of Support Vectors:  654
```

#### accuracy

99.1%

### combination #10

#### parameters

```txt
Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  30 
      gamma:  0.09 

Number of Support Vectors:  659
```

#### accuracy

100%
