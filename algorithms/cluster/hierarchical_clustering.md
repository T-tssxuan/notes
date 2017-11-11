## Hierarchical clustering
1. Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample. 
2. Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
3. Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
4. Average linkage minimizes the average of the distances between all observations of pairs of clusters.
5. FeatureAgglomeration，减少特征
6. Adding connectivity constraints(only adjacent clusters can be merged together), 维护局部结构
