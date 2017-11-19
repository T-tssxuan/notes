## Cluster

### 1. Robust Clustering as Ensemble of Affinity Relations
1. 常规的方法，假设所有的点一定会属于一些类别，这样简化了问题
2. 本文提出一种基于k-ary affinity relations的聚类算法，可以用于图或超图聚类算法
3. 通过KNN算法，初始化算法
4. 算法复杂度O(nthk)，t迭代次数，h点能包涵的超次个数，k类别多少

### 2. Fast Detection of Dense Subgraphs with Iterative Shrinking and Expansion
1. 近似mean shift algorithm
2. 高效检测dense subgraph; 相似性问题; 聚类
3. shrink phase and the expansion phase
4. 相似度计算，以及聚类分析

### 3. Dense Subgraph Partition of Positive Hypergraphs
1. positive hypergraphs: 除了自环都有正权重
2. 提出DSP框架，能够自动、精确、高效的检测密集子图
3. 提出计算DSP的算法
4. min-partition algorithm: 可以在常规PC上处理百万级数据
