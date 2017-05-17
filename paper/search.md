## Search related algorithm

#### 1. LSH(locality-sensitive hashing) Forest: Self-Tuning Indexes for Similarity Search
1. LSH 适应于相似度搜索
2. 基于LSH但是优化两点，a:消除LSH需要手动调节的缺点, b:改善LSH对于分布不均匀的数据消耗的平衡
3. LSH基本思想是使用特殊的locality-sensitive hash function，使得相似的项更加易容放在一起
4. 相似度查询主要集中在四个方面：Narest-neighbor queries, Duplicated detection, Link-based similarity search, Defining object representation
5. 使用树状索引在中低维向量中有效
6. 设计一个在动态数据集的相似度匹配索引算法
