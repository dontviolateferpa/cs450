library(datasets)


### AGGLOMERATIVE HIERARCHICAL CLUSTERING ###


# 01 - Load the dataset
myData = state.x77
summary(myData)

# 02 - Use hierarchical clustering to cluster the data on all attributes
#      and produce a dendrogram
## compute a distance matrix
distance = dist(as.matrix(myData))

## perform the clustering
hc = hclust(distance)

## plot the dendogram
plot(hc)

# 03 - Repeat the previous item with a normalized dataset and note any
#      differences
## scale the data
data_scaled = scale(myData)

## compute a distance matrix
distance = dist(as.matrix(data_scaled))

## perform the clustering
hc2 = hclust(distance)

## plot the dendogram
plot(hc2)


# 04 - Remove "Area" from the attributes and re-cluster (and note any differences)
data_scaled_s4 = data_scaled
data_scaled_s4$Area <- NULL

## compute a distance matrix
distance_s4 = dist(as.matrix(data_scaled_s4))

## perform the clustering
hc3 = hclust(distance_s4)

## plot the dendogram
plot(hc3)

# 05 - Cluster only on the Frost attribute and observe the results
data_scaled_s5 = data_scaled[, "Frost"]

## compute a distance matrix
distance_s5 = dist(as.matrix(data_scaled_s5))

## perform the clustering
hc4 = hclust(distance_s5)

## plot the dendogram
plot(hc4)

# note any differences
summary(hc)
summary(hc2)
summary(hc3)
summary(hc4)


### USING K-MEANS ###


## Cluster into k=5 clusters:
myClusters = kmeans(data_scaled, 5)

## Summary of the clusters
summary(myClusters)

## Centers (mean values) of the clusters
myClusters$centers

## Cluster assignments
myClusters$cluster

## Within-cluster sum of squares and total sum of squares across clusters
myClusters$withinss
myClusters$tot.withinss


## Plotting a visual representation of k-means clusters
library(cluster)
clusplot(data_scaled, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
