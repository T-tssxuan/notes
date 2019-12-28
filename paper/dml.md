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
