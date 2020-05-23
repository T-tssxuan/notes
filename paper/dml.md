## Distributed Machine Learning

### 1. A Survey on Distributed Machine Learning
- 使用GPU进行加速，另外还有很多基于ASIC的加速芯片
- 数据并行和模型并行
- 数据并行的假设是所有数据都是i.i.d.
- 机器学习算法分类
  - Feedback: Supervised learning, Unsupervised learning, Semi-supervised learning, Reinforcement learning
  - Purpose: Anomoly detection, Classification, Clustering, Dimensionality reduction, Representation learning, REgression
  - Mothod: Evolutionary algorithms(Genethic algorithms), Stochastic gradient descent based algorithms, Rule-based machine learning algorithms, Topic Models, Matrix Factorization
- Ensemble Method: Bagging, Boosting, Bucketing, Random Forests, Stacking, Learning Classifier Systems
- 常见的Topology：Tree, Ring, Parameter Server, P2P
- Communication: Bulk Synchronous Parallel(BSP), Stale Synchronous Parallel(SSP), Approximate Synchronous Parallel(ASP), Barrierless Asynchronous Parallel/Total Asynchronous Parallel(BAP/TAB)

### 2. ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
- 模型并性有较好的内存使用率，但是有通信效率较差；数据并行，有交好的通信效率，但是内存效率低
- 激活函数可以需要时再计算
- 混合精度训练
- 减少了内存占用，同时保持了通性效率

### 3. Mixed Precision Training
- 保存权重的FP32拷贝
- 对loss进行缩放
- 在FP32中使用FP16的算法
- 使用单精度拷贝，能够防止FP16精度不够，在乘以lr后直接变成0；另一方面，weight update也可能导致FP16不够用的情况
- 保留FP32的拷贝并不会消耗太多的内存，系统中主要的内存消耗在于act, fw, bw等，以及大batch size
- 由于一些较小的gradient十分有效，可能对其进行放大，然后在进行相关gradient操作之前进行缩小
- FP16可能产生regularizer的效果

### 4. Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes
- 结合使用mixed precision train
- LARS
- 高性能ring all reduce
- 在bias和bn上去掉weight decay
- 使用Tensor Fusion、Hierarchical All-reduce、Hybrid All-reduce加速ring reduce

### 5. AutoAugment: Learning Augmentation Strategies from Data
- 在数据增强花的时间并不多
- 实现在自动数据增强策略搜索
- 数据增强策略迁移
- 把数据增强转化成一个离散策略搜索问题
- 搜索空间：5个子策略，每个子策略包涵两个图片操作，每个操作包涵两个参数：1) 应用操作的概率，2）操作的量
- 离散化操作空间，形成搜索空间

### 6. Second Order Optimization Made Practical
- 现代优化器的一个重要挑战就是缩小理论和实际之间的差距
- 使用Schur-Newton algorithm可以把p'th root转化为一系列的矩阵乘法
- 一般情况，CPU属于空转状态，正好用于计算二阶能量

### 7. ReZero is All You Need: Fast Convergence at Large Depth, 2020.3
- 使用ReZero加快训练速度，且达到比LayerNorm和BatchNorm更高的水平
- 深层次的Transformer有严重的梯度爆炸和消失问题

### 8. Group Normalization, 18.03
- LN在RNN/LSTM等效果好，IN在GAN效果好，但是在图像作用都有限
- 主要缓解BN在小batch size时效果的变差
- GN更适应于图像

### 9. Memory-Efficient Adaptive Optimization, 19.01
- 很多梯度都有相关性，input和output
- 
