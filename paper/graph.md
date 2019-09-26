## Graph

### DEEP GRAPH INFOMAX, 2018
- 相对以前的结果，DGI不依赖于random walk
- 在一些图分类算法试用，结果甚至超过了一些监督学习
- 首先学习一些兴趣点的子图，然后再用于下一层级
- random wolk过份强调邻近关系，而且也不知道选取的边是否有用
- Contrastive method: 训练encode来突出相关与非相关

### PYTORCH-BIGGRAPH- A LARGE-SCALE GRAPH EMBEDDING SYSTEM
- 借鉴了starspace
- 在GCN中，大部分结点是已经特征化了
- 无效的edge是无效的

### Graph Neural Networks: A Review of Methods and Applications
- graph embedding的问题如下：无参数共享、泛化能力差、无法处理动态变化的图
- 目前没有好的方法来处理动态图谱
- 可以用于很多地方，监督、半监督、无监督、强化学习等

### Translating Embeddings for Modeling Multi-relational Data
- 提出一种简单而有效的图处理方法
- 使用l2-norm

### Node2vec: Scalable feature learning for networks

### Deepwalk: Online learning of social representations

### Line: Large-scale information network embedding


