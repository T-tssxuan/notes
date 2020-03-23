## Neural Architecture Search

### 1. Neural Architecture Search- A Survey
- search space, search strategy, performance estimation strategy
- search space: linear or combine, keep dimension or reduce dimension cells, meta-architecture is a part of NAS
- search strategy: random search, Bayesian optimization, evolutionary methods, reinforcement learn- ing (RL), and gradient-based methods

### 2. Learning Transferable Architectures for Scalable Image Recognition
- 使用搜索cell的方法来搜索更好的网络结构
- 使用5步搜索法来构造网络

### 3. AutoML-Zero: Evolving Machine Learning Algorithms From Scratch
- 从零开始进行计算
- 三种变种
  - 随机插入
  - 随机变换整个组件的操作
  - 随机修改参数
- 加速
  - 检测相同的算法
  - 使Evolved Transformer

### 4. The Evolved Transformer, 2019.01
- 常见的方法主要的问题有：
  - 很难热启动Transformer
  - ENAS和DARTS需要很多内存
  - 在视频领域最好的架构来自于进化算法

### 4. Random Search and Reproducibility for Neural Architecture Search
