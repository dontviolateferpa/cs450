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


# 01 - 03
## Cluster into k=5 clusters:
myClusters = kmeans(data_scaled, 3)

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

error = NULL
for (i in 1:25) {
  tempClusters = kmeans(data_scaled, i)
  error[i] = tempClusters$tot.withinss
}
plot(error)

# 04 - Evaluate the plot from the previous item, and choose an appropriate k-value using
#      the "elbow method" mentioned in your reading. Then re-cluster a single time using
#      that k-value. Use this clustering for the remaining questions.

## Cluster into k=5 clusters:
myClusters_elbow = kmeans(data_scaled, 5)

## Summary of the clusters
summary(myClusters_elbow)

## Centers (mean values) of the clusters
myClusters_elbow$centers

## Within-cluster sum of squares and total sum of squares across clusters
myClusters_elbow$withinss
myClusters_elbow$tot.withinss

## Plotting a visual representation of k-means clusters
clusplot(data_scaled, myClusters_elbow$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

## Cluster assignments (list the state in each cluster)
myClusters_elbow$cluster



### ABOVE AND BEYOND ###
dataVolcano = volcano
summary(dataVolcano)

## scale the data
data_scaled_volc = scale(dataVolcano)

## compute a distance matrix
distance_volc = dist(as.matrix(data_scaled_volc))

## perform the clustering
hc_volc = hclust(distance_volc)

## plot the dendogram
plot(hc_volc)


error_volc = NULL
for (i in 1:25) {
  tempClusters = kmeans(data_scaled_volc, i)
  error_volc[i] = tempClusters$tot.withinss
}
plot(error_volc)

myClustersElbowVolc = kmeans(data_scaled, 4)

clusplot(data_scaled_volc, myClustersElbowVolc$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
