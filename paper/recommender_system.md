## Recommender system related paper

#### 1. Collaborative Filtering for Implicit Feedback Datasets
1. 把数据样本当成广义上的正例和负例，这样构成了一个关于隐性反馈的因果模型
2. CF存在冷启动的问题
3. 无负反馈数据、本质上存在噪声、显反馈的数据代表的是喜好，而隐反馈数量代表是置信度、衡量隐反馈系统需要相应的措施。
4. 前期的CF算法主要是面向用户的，后期的是面向ITEM的
5. item-oriented model 都有没有很好的区别用户偏好和置信度
6. 使用交叉计算user-factor和iter-factor的方法来计算矩阵

#### 2. Deep Neural Networks for YouTube Recommendations
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

#### 3. Collaborative Metric Learning
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

#### 4. Combining Collaborative Filtering with Personal Agents for Better Recommendations
1. 使用IF和CF结合达到更好的过滤目标
2. CF的两个主要目标：哪个是我喜欢的，我有多喜欢某件物品

#### 5. Recurrent Recommender Networks
1. 不使用隐状态，只学习转换函数
2. 一般的方法考虑不到时序因素：1.人们对电影的看法是随时间改变的，2.对电影的感观随季节改变，3.用户的兴趣也是在不断的改变的
3. 一些方法也有违因果关系，其使用未来的结果来衡量现在的情况
4. 一个抓住要点的模型需要同时考虑时序因果关系、用户对电影的评价等因素
5. 使用离散的隐状态进行描述
6. 使用auto-encoder来进行非线性转换
7. 同时考虑随时间变化的特性，以及不变特性
8. Hedonic adaptation，用户在看到更喜欢的电影后，对过去的喜欢的电影的评介分下降
9. 不能使用常规的BP，而是user和movie交替执行，使用subspace descent.
