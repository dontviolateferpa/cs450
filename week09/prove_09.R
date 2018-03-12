library(arules)
g_data <- data(Groceries)
summary(g_data)

# The 5 rules you can find with the highest support
rules_01 <- apriori(Groceries, parameter = list(support = 0.00233858668022369))
summary(rules_01)
inspect(rules_01)

# The 5 rules you can find with the highest confidence
rules_02 <- apriori(Groceries, parameter = list(confidence = 0.1395))
summary(rules_02)
inspect(rules_02)

# The 5 rules you can find with the highest lift
rules_03 <- apriori(Groceries, parameter = list(support = 0.001))
summary(rules_03)
inspect(rules_03)
write(rules_03,
      file = "C:\\Users\\Zachary\\OneDrive\\Documents\\09 BYUI 2018 Winter (Override)\\CS 450 - Machine Learning and Data Mining\\cs450\\week09\\association_rules.csv",
      sep = ",",
      quote = TRUE,
      row.names = FALSE)

rules_04 <- apriori(Groceries, parameter = list(support = 0.001, confidence = 0.01))
summary(rules_04)
inspect(rules_04)

rules_05 <- apriori(Groceries, parameter = list(support = 0.001))
summary(rules_05)

