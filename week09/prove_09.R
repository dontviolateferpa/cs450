library(arules)
g_data <- data(Groceries)
summary(g_data)

rules_01 <- apriori(Groceries, parameter = list(support = 0.001))
summary(rules_01)
