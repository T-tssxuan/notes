## Search related algorithm

#### 1. LSH(locality-sensitive hashing) Forest: Self-Tuning Indexes for Similarity Search
1. LSH 适应于相似度搜索
2. 基于LSH但是优化两点，a:消除LSH需要手动调节的缺点, b:改善LSH对于分布不均匀的数据消耗的平衡
3. LSH基本思想是使用特殊的locality-sensitive hash function，使得相似的项更加易容放在一起
4. 相似度查询主要集中在四个方面：Narest-neighbor queries, Duplicated detection, Link-based similarity search, Defining object representation
5. 使用树状索引在中低维向量中有效
6. 设计一个在动态数据集的相似度匹配索引算法

#### 2. An Investigation of Practical Approximate Nearest Neighbor Algorithms
1. Question: can earlier spatial data structure approaches to exact nearest neighbor, such as metric trees, be altered to provide approximate answers to proximity queries and if so, how?
2. 介绍一种新的度量树，这种树允许一些结点同时出现在子结点和父结点
3. 在中等大小，kd-tree, mtric tree常用，在高维下，metric tree, ball-tree也常常用到
4. metric tree是一种可以快速查找的最近邻居的算法，其使用一稀疏的层级结构来组织结点
5. split tree是一种metric tree变种，其使用软边界，而不是硬边界
6. 使用split tree的defeatist search策略，可以避免mt-df中的backtrack问题
7. 为了平衡分布，当一个子结点占比大于70%时，我们退化为metric-tree的结点
