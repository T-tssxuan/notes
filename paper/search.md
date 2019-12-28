## Search related algorithm

### 1. LSH(locality-sensitive hashing) Forest: Self-Tuning Indexes for Similarity Search
1. LSH 适应于相似度搜索
2. 基于LSH但是优化两点，a:消除LSH需要手动调节的缺点, b:改善LSH对于分布不均匀的数据消耗的平衡
3. LSH基本思想是使用特殊的locality-sensitive hash function，使得相似的项更加易容放在一起
4. 相似度查询主要集中在四个方面：Narest-neighbor queries, Duplicated detection, Link-based similarity search, Defining object representation
5. 使用树状索引在中低维向量中有效
6. 设计一个在动态数据集的相似度匹配索引算法

### 2. An Investigation of Practical Approximate Nearest Neighbor Algorithms
1. Question: can earlier spatial data structure approaches to exact nearest neighbor, such as metric trees, be altered to provide approximate answers to proximity queries and if so, how?
2. 介绍一种新的度量树，这种树允许一些结点同时出现在子结点和父结点
3. 在中等大小，kd-tree, mtric tree常用，在高维下，metric tree, ball-tree也常常用到
4. metric tree是一种可以快速查找的最近邻居的算法，其使用一稀疏的层级结构来组织结点
5. split tree是一种metric tree变种，其使用软边界，而不是硬边界
6. 使用split tree的defeatist search策略，可以避免mt-df中的backtrack问题
7. 为了平衡分布，当一个子结点占比大于70%时，我们退化为metric-tree的结点

### 3. Product quantization for nearest neighbor search
- 把空间分解成为低维度的子空间，作为子空间的笛卡尔集
- 把向量聚焦到几个指定的中心，进行向量量化
- 进行pq量化，空间可以扩展到k^m
- 每个量化都进行正交
- 在ADC和SDC两个设定中，计算开销差不多，但是他们之间的精度有所差距
- 可以在ADC的基础上进行修正，但是修正版只是距离数值更准确，但KNN的精度不如末修正的版本
- 通过IVF进行加速
- 针对每个item的，不但存储了其索引，也存储了残差
- x的最近临和其量化值中心通常不在同一个中心

### 4. Video Google: A Text Retrieval Approach to Object Matching in Videos
- 参考文本检索中相关信息，进行视频检索
