library(arules)
g_data <- data(Groceries)
summary(g_data)

# The 5 rules you can find with the highest support
rules_01 <- apriori(Groceries, parameter = list(support = 0.00233858668022369))
summary(rules_01)

rules_02 <- apriori(Groceries, parameter = list(support = 0.001))
summary(rules_02)

rules_03 <- apriori(Groceries, parameter = list(support = 0.001))
summary(rules_03)

rules_04 <- apriori(Groceries, parameter = list(support = 0.001))
summary(rules_04)

rules_05 <- apriori(Groceries, parameter = list(support = 0.001))
summary(rules_05)

