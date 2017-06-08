## GAN related paper

#### 1. Generative Adversarial Nets
1. Precision: The fraction of detections reported by the model that were correct.
2. Recall: The fraction of true events that were detected.
3. F-score: 2pr/(p + r).
4. Sometime add a small fraction of the total number of example will not have a noticeable impact on generalization error. It is therefore recommended to experiment with training set sizes on a logarithmic scale.

#### 2. InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
1. 非监督学习存在一个缺点，他是后续操作并不知道非监督学习的结果具体表示什么，而如果能够对每个表示进行合理的解释，这将十分有益。
2. InfoGAN可以提取离散和连续的隐藏信息，并且不会花很大的代价
3. 引入两个噪声，z是不右解释的噪声，c针对一些突出点的噪声
4. 隐变量对信息有很大的作用，也就是说，如果给定c，可以很大的降低p(x)的熵
5. 对离散变量，我们使用softmax计算，对于连续变量，我们使用posterior计算
6. 用抽象来教机器

#### 3. Adversarial Multi-task Learning for Text Classification
1. 使用神经网络可以提取不变特征，但是由于其可能被其它的任务影响，可能结果不好，本文提出一种使用gan来降低这种影响。
2. 多任务学习是一种改进单任务的一种方法
3. 对于share-private，使用GAN可以更干净,GAN使用内在的正交来防止他们之间相互影响
4. 使用adversarial training and orthogonality constraints来隔离影响，使用GAN来学习TASK之间共有的特性，使用orthogonality去除privata和shared中多余的特征
5. 可以使用无标签计算
6. 一个简单的推论，好的共享特征应该有更多共同特征
7. 使用对称网络，使得系统不能分辩句子到底是来自于哪一个网络
8. 损失函数定义为，错分误差，ADV误差，以及正交误差

#### 4. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
1. 我们的目标是训练G: X - Y，并且通过F: Y - X进行对偶训练，最后最小化由F(G(X)) = X完成训练目标
2. 因为数据集的有限，寻求一种算法，可以自动学习不同的数据集之间的不同，以及关联
3. 不像之前的任何方法，本算法不依赖任何既定的特征或者相似函数

#### 5. LEARNING TO PROTECT COMMUNICATIONS WITH ADVERSARIAL NEURAL CRYPTOGRAPHY
1. 

#### 6. Improved Techniques for Training GANs
1. Feature matching用于防止Generator不会被overtrain
2. GAN失败的主要原因是genrator塌缩到一点
3. minibatch discrimination的主要思想是，使用多个样本结合，而不是单个的，这样可能可以防止collapse
4. virtaul batch normalization(VBN)

#### 7. IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models
1. 信息获取主要有两种模式：a. 预测给出的文档相关性；b. 给出文档对，判别他们之间的相关性
2. 判别模型：挖掘标注与非标注的数据中的信息，用于指导训练拟合文档内在相关性分布生成模型
3. 生成模型：生成判别模型难以判别的例子
4. 经典相关性模型着重于，如何从查询生成(相关)文档；独立模型，每个token是独立从相关文档档中生成；统计语言模型一般是从文档中生成查询元素；在词嵌模型中，词从其上下文中生成；在推荐系统中，也有类似的方法，从item的上下文中生成item
5. 模型扩展到pointwise, pairwise, listwise, 其中pointwise基于人的判断来衡量相关性，pairwise主要是在所有文档对中找出最相关的文档对，listwise着重于返回最合理的相关性排序
6. 观察到的正例和未观察到的正例之间会存在内在联系，生成器需要基于判别器的信息来快速推动这些未观察到的正例
7. 与conditional GAN有些相似
8. 生成模型提供了一种新的负采样方式
9. 使用IR的奖励机制，是在传统模型中不可获取的
10. 应用于：网页搜索在线排序(sf: LambdaRank, LambdaMART, RankNet)、item推荐系统(cf matrix factorisation)、问答系统(退化成IR的评估)

#### 8. DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data
1. 我们把潜在生成空间重新参数化生成一混合模型，并把模型与GAN一同训练
2. 为了衡量类内的多样性，提出一种"inception-score"，这个评估方法与人类评估行为有一定联系
3. Gaussian 成员之间是正交的
4. 
