## Recommender system related paper

### 1. Collaborative Filtering for Implicit Feedback Datasets
1. 把数据样本当成广义上的正例和负例，这样构成了一个关于隐性反馈的因果模型
2. CF存在冷启动的问题
3. 无负反馈数据、本质上存在噪声、显反馈的数据代表的是喜好，而隐反馈数量代表是置信度、衡量隐反馈系统需要相应的措施。
4. 前期的CF算法主要是面向用户的，后期的是面向ITEM的
5. item-oriented model 都有没有很好的区别用户偏好和置信度
6. 使用交叉计算user-factor和iter-factor的方法来计算矩阵

### 2. Deep Neural Networks for YouTube Recommendations
1. Scale: The system get a very large scale of data
2. Freshness: There are a very dynamic corpus with many hours of video are uploaded per second.
3. Noise: Historical user behavior on YouTube is inherently difficult to predict due to sparsity and a variety of unobservable external factors.
4. The candidate generation network only provides broad personalization via collaborative filtering.
5. 在线上使用A/B test进行评估模型
6. 深度学习的主要任务是基于用户history和context学习到用户的表示
7. 使用负采样进行加速学习
8. 发现使用何种邻近搜索算法影响并不大
9. 直拉把各表示平均后进行叠加
10. 重现用户已浏览信息，效果非常差
11. 许多CF选择一个标签或者上下文做为随机生成用户历史中没有的内容的候选
12. 引入feature的平方，二次方做为输入
13. weighted logistic regression
14. 加入了视频的时间标记

### 3. Collaborative Metric Learning
1. 提出一种CML，不仅依赖于user preference，而且考虑user-user，item-item等因素
2. 使用OFF-THE-SHELF、近似最近邻居搜索加快了TOP-K推荐任务的速度
3. 三角不等式也是十分重要的
4. 大多数据点乘矩阵不能满足三角不等式
5. 一般的衡量矩阵不能反映user-user和item-item的关系
6. 全局最优的本质是使相似的距离更近，不想似的距离更远
7. 隐反馈存在一数据不平均的偏差
8. 使用weighted ranking loss
9. Weighted Approximate-Rank Pairwise(WARP) loss
10. Using negative sample mining
11. 防止数据稀疏，并使用covariance规范化

### 4. Combining Collaborative Filtering with Personal Agents for Better Recommendations
1. 使用IF和CF结合达到更好的过滤目标
2. CF的两个主要目标：哪个是我喜欢的，我有多喜欢某件物品

### 5. Recurrent Recommender Networks
1. 不使用隐状态，只学习转换函数
2. 一般的方法考虑不到时序因素：1.人们对电影的看法是随时间改变的，2.对电影的感观随季节改变，3.用户的兴趣也是在不断的改变的
3. 一些方法也有违因果关系，其使用未来的结果来衡量现在的情况
4. 一个抓住要点的模型需要同时考虑时序因果关系、用户对电影的评价等因素
5. 使用离散的隐状态进行描述
6. 使用auto-encoder来进行非线性转换
7. 同时考虑随时间变化的特性，以及不变特性
8. Hedonic adaptation，用户在看到更喜欢的电影后，对过去的喜欢的电影的评介分下降
9. 不能使用常规的BP，而是user和movie交替执行，使用subspace descent.

### 6. Scalable Coordinate Descent Approaches to Parallel Matrix Factorization for Recommender Systems
1. [Code link](http://www.cs.utexas.edu/~rofuyu/libpmf/)
2. Alternating Least Squares和Stochastic Gradient Descent是两个很常见的用于矩阵分解的方法
3. 提出一个CCD++矩阵分解方法
4. CCD++可以方便的适应于内存共享、多核和分布式系统
5. ALS算法，更多的问题是分布式计算时，内存超出
6. SGD时间复杂度更好，但是并行可能存在重写问题
7. 使用coordinate descent每次更新一个变量
8. CCD++ in Multi-core Systems
9. CCD++ in Distributed Systems
10. 由于W和H的量大，以及Hessian矩阵计算最也大，ALS不适合并行
11. DSGD可以进行并行处理

### 7. Low-Rank Linear Cold-Start Recommendation from Social Data
1. LOCO a.使用线性回归获取对于偏好最优的社交信息，b.低维的权值克服社交数据高维特性，c.可扩展的low-rank权值，全名用randomised SVD
2. 冷启动问题，可以使用一些附加信息来完成，如人口统计信息
3. 寻找一个隐变量空间U，可以同时来预估用户偏好和社交特质
4. 寻找T，使得社交属性X可以拟合用户属性
5. social neighbourhood model没有考虑了属性之间的相关性，存在underfit风险；bpr-linmap模型考虑到，但是存在overfit的风险；CMF模型，计算复杂
6. 为了使了回归可以训练，对W的rank进行限制，这样也带了等式变成non-convex的问题

### 8. Getting Deep Recommenders Fit: Bloom Embeddings for Sparse Binary Input/Output Networks
1. 来自推荐领域的数据，常常具有很高维的输入输出特征，这样导致非常难以训练
2. 使用Bloom embedding压缩输入输出的高维编码，在保证精度的条件下压缩至原有数据的1/5，在一些情况下甚至提升精度
3. 使用Bloom embedding(BE)对输入输出进行编码
4. 最后输出层使用softmax与原始空间建立联系
5. 主要关注在推荐和协同过滤，但在自然语言处理也可以有应用
6. 在数据ML(Movielens), MSD(Million song data set), AMZ(Amazon book reviews), BC(Book Crossing), YooChoose(YC), Penn treebank(PTB), CADE(CADE web directory)进行试实验
7. 对比HT(Hashing trick), ECOC(Error-correcting output codes), PMI(Pairwise mutual information), CCA(Canonical correlation analysis)，BE拥用即时操作、常数时间复杂度、无监督等优点
8. 当m接近d时，score并没有减少，证明BE没有弱化结果
9. 可能在SVD相关上，有更好的结果
10. CBE(co-occurrence-based Bloom embedding)，使用共现矩阵来降低冲突

### 9. Multi-Rate Deep Learning for Temporal Recommendation
1. 使用用户近期数据进行预测，但是可能出现冷启动wenti
2. Long term and short term分别对用户进行建模
3. Deep Semantic Structured Model 用户静态兴趣，LSTM用户week or daily, LSTM全局兴趣

### 10. Wide & Deep Learning for Recommender Systems
1. 使用乘法生成的特征能够很好的表现数据情况，但是其需要特征工程
2. 当交互网络过于稀疏时，很可能推荐出关系不大的数据
3. 推荐引擎，memorization和generalization
4. 使用embedding可能导致过分generalize，从而使得推荐很多非相关的数据
5. FTRL for wide model, AdaGrad for deep model
6. 每一批数据到达，都需要重新进行训练，很慢，使用热启动

### 11. Use of Deep Learning in Modern Recommendation System: A Summary of Recent Works
1. 三个主要的分类：collaborative filtering, content based, hybrid recommendation models
2. 应用到collaborative filtering的占大多数

### 12. SAR: Semantic Analysis for Recommendation
1. 使用更丰富的潜在特征进行推荐
2. 把user和item表示在同一个空间
3. 使用2层体系
4. 不推荐

### 13. Deep Reinforcement Learning for List-wise Recommendations
1. 大部分推荐系统最大化短期收益，但最大化长期收益可以取得更多利润
2. a.跟用户交互时不断更新策略，直到收敛，b. 最大化长期收益
3. DQN可以推荐list，但是其忽略了item之间的关系
2. 当忽略item之间的相关性时，推荐的结果非常相似，但是一般来说，互补的推荐比相似性推荐可以得到更高的回报。
3. 提出一种主要方法，获取相互关系，并生成互补队列
4. 提出的方法，即可以适应large action space，也可以减少冗余，加快计算
5. 使使模拟的方法进行线下训练，
6. 基于用户行为，训练item的低维表示
7. 根据CF，相似的用户会有相同的选择，在历史中匹配相同的状态和行为用户模拟
8. 使用状态和行为的相似度来衡量回报权重，为了使搜索空间变小，从回报空间进行计算
9. 使用序列长度控制短期和长期行为

### 14. Deep Learning based Recommender System: A Survey and New Perspectives
1. 推荐系统主要有：cf, content-based, hybird recommender. 但是这些方法在稀疏数据和冷启动方面存在局限性
2. rating prediction, ranking prediction
(top-n recommendation) and classifcation. 
3. 大部分是转会为排序问题，一部分是为评分问题，少部分转化为分类问题
4. NCF，传统CF可以视为NCF的特例，NCF可以使用Negative smapling进行加速
5. DW可以用于解决回归和分类模型
6. DeepFM 是一个端到端的模型，其包涵了Factorization machine和MLP
7. Attention 可以提升推荐效果，一般分为item-level和component-level
6. Autoencoder: a) 使用瓶颈层提取信息，b) 完善用户评分
8. pairwise model is more suitable for ranking lists generation
9. CNN
10. RNN: a) pretraining with full data, fine-tuning with the most recent click-sequences, b) consider the side information, c) 
11. RBM
12. NADE
13. GAN
14. Apart from accuracy, other evaluation metrics such as, diversity [3, 102], novelty, serendipity, coverage, trustworthiness, privacy, interpretability etc. 

### 15. Learning Continuous User Representations through Hybrid Filtering with doc2vec
1. user2vec，使用用户历史行为转化成向量
2. context2vec，用户和app metadata转化成向量

### 16. Reinforcement Learning based Recommender System using Biclustering Technique
1. 把推荐转化成MDP过程，但是存在过多的离散行为
2. 使用biclustering转化成gridworld game

### 17. Demystifying Core Ranking in Pinterest Image Search
1. cascading for acceration online performance

### 18. Evaluation of Session-based Recommendation Algorithms
1. session-based algorithms predict user immediate next actions
2. amazon 'user buy also buy' can be recognized a extreme case of such a session-based approach.
3. AR method count the co-occurrency of i and j in a session
4. using different nearest neighbor schemes
5. gru4rec: BPR loss, TOP1 loss, MAX loss(best)
6. Factorization-based Methods: Factorized Personalized Markov Chains (fpmc), Factored Item Similarity Models (fism), Factorized Sequential Prediction with Item Similarity Models (fossil), Session-based Matrix Factorization (smf).
7. Popularity bias
8. Data set: e-commerce, music, news.
9. 主要对四种算法进行比较: baseline methods, nearest-neighbor techniques, recurrent neural networks, and (hybrid) factorization-based methods.

### 19. Analyzing and Characterizing User Intent in Information-seeking Conversations
1. CA for simple question are typically referred to as information-seeking conversations
2. Gain insight on human intent dynamics during information-seeking conversations.

### 20. Collaborative Memory Network for Recommendation Systems(pass)
1. Proposed Collaborative Memory Networks(CMN), a network unify global(latent factor) and local(neighborhood-based)
2. memory component and neural attention mechanism -> neighborhood component; stacking multiple memory modules yield deeper architectures.
3. CF can generally be grouped in three categories: memory or neighborhood-based approaches, latent factor models and hybrid models.

### 21. An Universal Image Attractiveness Ranking Framework
1. combine dnn and rank net gets deep attractiveness rank net.
2. constructed a data set IASBS(image attractiveness side-by-side)
3. 
3. using in commercial search engine.

### 21. Cross Domain Regularization for Neural Ranking Models using Adversarial Learning
1. tranditional model made less assumption about the data, which get more robust cross domain result
2. Our goal is to train deep neural ranking models that learn useful
representations from the data without “overfitting” to the distributions of the training domains.
3. using domain adversarial discriminator to restrict the ranking model.
4. We propose using an adversarial agent to force the features learned by the ranking model to be domain agnostic by shifting the model parameters in the opposite direction to domain specific spaces on the manifold

### 22. MARS: Memory Attention-Aware Recommender System
1. MARS utilizes a memory component and a novel attentional mechanism to learn deep adaptive user representations
2. To answer this question, we observed one essential characteristic of users’ interests: diversity
3. We argue that a fixed user representation largely restrains models from accurately modeling users’ diverse interests
4.  a memory component and a novel item-level attention mechanism
5. Adaptive user representation, and using memory component to capture users' interest
6. An interpretable model
7. Strong Performance

### 23. Deep Interest Network for Click-Through Rate Prediction
1. 使用定长的vector会使得模型不能理解用户行为的多样性
2. mini-batch aware regularization and data adaptive activation function which can help training industrial deep networks with hundreds of millions of parameters.
3. we develop a novel mini-batch aware regularization where only parameters of non-zero features appearing in each mini-batch par- ticipate in the calculation of L2-norm, making the computation acceptable. 
4. 用户可能有多角度的兴趣，但只有部分兴趣被激活
5. Dice with moving mean and variance

### 24. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
1. Wide & deep architecture of DeepFM. The wide and deep component share the same input raw feature vector, which enables DeepFM to learn low- and high-order feature interactions simulta- neously from the input raw features.
2. FM一般只考虑到order-2
3. It can capture order-2 feature interactions much more effec- tively than previous approaches especially when the dataset is sparse.
4. 手特征工程

### 25. Deep Reinforcement Learning for Page-wise Recommendations, 2018
1. 利用用户实时反馈
2. 如何合理的展示一页的商品
3. 一种生成相容的候选并在2-D的页面展示的方法
4. 在真实的电子商务数据中应用
5. Actor: Actor framework with the Encoder-Decoder architecture.
6. 使用Encoder生成初始状态和实时用户状态
7. 对商品在2D空间上进行排列，并使用CNN
8. 把卷积结果输入到RNN中，并以sinit作用RNN初始状态，并对卷积结果使用attention
9. 使用Decoder进行状态空间生成
10. 使用cosine similarity把生成的item映身至真实空间

### 26. Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning
1. static and short term reward
2. 认识到负反馈的重要，并使用其增强推荐效果
3. 正样本可能会被负样本给覆盖
4. 引入负状态序列
5. 针对正样例选择负样例，三个原则：同类，不同值，时间近
6. Reinforcement learning for robots using neural networks

### 27. Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application
1. 提出使用rl为最大化多阶段搜索一个session中的累积收益
2. 搜索引擎常常需要进行多轮交互
3. 证明了最大长累积收益是有必要的
4. discount与搜索引擎是否要考虑长期收益
5. 证明了最大化累积收益就是最大化GM

### 28. Online Learning to Rank in Stochastic Click Models

### 28. Stochastic Rank-1 Bandits

### 29. Multiple-play bandits in the position-based model

### 26. Session-based recommendations with recurrent neural networks
1. ***这里说gru好于lstm***

### 25. Ad click prediction: a view from the trenches

### 26. Deep Crossing: Web-scale modeling without manually crafted combinatorial features

### 27. Pairwise Interaction Tensor Factorization for Personalized Tag Recommendation

### 28. Learningandadaptivityininteractive recommender systems

### 29. A hybrid web recommender system based on q-learning.

### 30. Cumulated gain-based evaluation of IR techniques

### 30. A user browsing model to predict search engine click data from past observations

### 24. Entire Space Multi-Task Model

### 19. Sequence-Aware Recommender Systems 
1. 

### A survey of collaborative fltering techniques

###  Cross-domain recommender systems: A survey of the state of the art.

### Cross Domain Recommender Systems: A Systematic Literature Review

### 15. Embedding-basedNewsRecommendationforMillionsofUsers.

### 15. LocallyConnectedDeepLearningFrameworkfor Industrial-scale Recommender Systems

### 14. Linear Submodular Bandits and their Application to Diversified Retrieval

### 15. Neural word embedding as implicit matrix factorization.

### 16. Personalized Deep Learning for Tag Recommenda- tion

### 10. A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems
1. 

### Author Commitment and Social Power: Automatic Belief Tagging to Infer the Social Context of Interactions

### A quantum-inspired classical algorithm for recommendation systems

### Optimizing search engines using clickthrough data

### Learning to rank for information retrieval.

### Usingconfidenceboundsforexploitation-explorationtrade-offs

### Cas- cading bandits: Learning to rank in the cascade model


### The epoch-greedy algorithm for multi-armed bandits with side information.

### Interactively optimizing information retrieval systems as a dueling bandits problem

### Ranked bandits in metric spaces: learning diverse rankings over large document collections
