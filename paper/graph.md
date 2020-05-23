## Graph

### 1. DEEP GRAPH INFOMAX, 2018
- 相对以前的结果，DGI不依赖于random walk
- 在一些图分类算法试用，结果甚至超过了一些监督学习
- 首先学习一些兴趣点的子图，然后再用于下一层级
- random wolk过份强调邻近关系，而且也不知道选取的边是否有用
- Contrastive method: 训练encode来突出相关与非相关

### 2. PYTORCH-BIGGRAPH- A LARGE-SCALE GRAPH EMBEDDING SYSTEM
- 借鉴了starspace
- 在GCN中，大部分结点是已经特征化了
- 无效的edge是无效的

### 3. Translating Embeddings for Modeling Multi-relational Data
- 提出一种简单而有效的图处理方法
- 使用l2-norm

### 4. Graph Neural Networks: A Review of Methods and Applications
- graph embedding的问题如下：无参数共享、泛化能力差、无法处理动态变化的图
- 目前没有好的方法来处理动态图谱
- 可以用于很多地方，监督、半监督、无监督、强化学习等
- GCN, GAT, GGNN
- GNN主要针对三个方面，分类、链接预测、聚类等

### 5. Deep Learning on Graphs: A Survey
- 图数据，相对比较不规则
- GNN和GCN主要处理一些end2end的任务，GAE主要处理表征学习
- 因为图的特性，CNN的常操作不能应用到图上
- 添加一个结节，做为全局结点

### 6. A Comprehensive Survey on Graph Neural Networks
- 新的分类体系：a. 递归GNN, b. 卷积GNN, c. 图autoencoder, d. 时空GNN

### 7. Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs
- 主要特点：
    - 使用graph抽象概念
    - 灵活支持任意方式传送信息的API
    - 支持很大的动态图
    - 高效的内存利用和训练速度

### 8. Benchmarking Graph Neural Networks
- 在DGL的基础上测试
- 提供中等数据集
- 在同等的预算下进行比较
- MLP在小数据集上不比GNN网络差
- 在大数据集上，GNN能比MLP好得多
- GCN依赖于同性数据，导致无区别网络结构的能力
- Residual connection可以改善结果
- Normalize层可以改善结果
- l 1 

### 9. Graph Convolutional Matrix Completion
- 基于graph-based auto-encoder构建user和item的隐式特征，然后用这些特征去预测user和item之前的link
- 这种方式对于有外部信息的推荐十分直接，特别是如社交网络等
- 多层叠加赶不上单层
- 不是所有的评价都有一样的频率，所以可以引入权重共享

### 10. Inductive Matrix Completion Based on Graph Neural Networks, 2020
- 可以不使用外部信息
- 局部信息可以非常有效
- 长范围依赖可能不必要
- 以前的都是学习node表示，本文学习subgraph表征
- 使用跳数对不同的label进行编码
- 在学习时，item和user都使用各自的关系子图

### 11. Semi-Supervised Classification with Graph Convolutional Networks, 2017
- 改造计算，引入GCN
- 使用切比雪夫多项式进行定义k-order邻居
- 为了防止抖动，重新定义拉谱拉斯乘子

### 12. Wavelets on Graphs via Spectral Graph Theory, 2009
- wavelet具有强大的不变性

### 13. Spectral Networks and Deep Locally Connected Networks on Graphs, 2014
- 提出两种方法：a. 基于领域层级聚类，b. 基于拉谱拉斯频谱
- 基于权重的局域化网络，且只关注邻居，使得复杂度降为：O(Sxn)

### 14. Graph attention networks, 2017.10
- 使图模型能够进行传导和归纳
- 使用avg而非concat对multi-head做处理

### 6. Spectral Networks and Locally Connected Networks on Graphs
- 

### 7. Laplacian eigenmaps and spectral techniques for embedding and clustering

### 8. The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains

### 9. Geometric deep learning: going beyond Euclidean data

### 10. Graph attention networks

### Node2vec: Scalable feature learning for networks

### Deepwalk: Online learning of social representations

### Line: Large-scale information network embedding


