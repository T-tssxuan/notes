# GAN related paper

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

#### 9. ENERGY-BASED GENERATIVE ADVERSARIAL NETWORKS
1. 使用energy-based区分函数，可以让我们应用到更多的架构，而不仅仅是二分的logistic输出
2. 展示了一基于auto-encoder的energy-based架构，energy做为重构误差，结果显示能比一般的GAN得到更好的结果
3. energy-based是一个数据驱动过程，在能量面上，正确的数据在较低的能量点，错误的数据在高能量点
4. 区分器可以看成是生成器可训练的cost function，其在真实数据聚集区域能量低，在其它区域能量高；生成器可以看成是cost function可训练的参数，其目标是尽量降低cost function的值
5. 由公式推导可以知道，最后的状态为：Pg = Pdata，达到了纳什均衡，并且用D(x)几乎在所有位置小于m
6. 使用autoencoder的两个好处，其一，相对于单比特区分，训练更有效；基二，energy-based方法常常使用auto-encoder，另外其还有无监督等优点
7. repelling regularizer for autoencoder: 用于防止判别autoencoder塌缩到一点或者一些数据上，类似于《Improved techniques for training gans》中的"minibatch discrimination"
8. “repelling regularizer”，使用cosin距离而非euclidean距离，是使其对值大小不敏感，并且PT(Pulling-away Term)定义的是生成器的Loss而非判别器

#### 10. Adversarially Regularized Autoencoders for Generating Discrete Structures
1. 结合离散autoencoder和编码空间GAN，ARAE(adversarially regularized autoencoder)
2. 使用GAN生成器输出连续编码空间表示，GAN判别器控制结果，使用autoencoder解码器解码把连续空间编码解析成离散编码
3. 本论文的生成器与WGAN的主要区别在于，GAN不在是基于原始数据集进行生成，而是在autoencoder的编码空间进行生成
4. 训练过程：a训练AE，b使用样本(正例)训练判别器和编码器、生成样本(负例)训练判别器，c使用判别器训练生成器
5. 关于AE训练：autoencoder单独训练是为了训练其编码和解码能力，可以说是面向解码器训练编码器; AE编码器与GAN判别器一起训练，是使用判别器来规范化编码器，或者说是面向判别器特征空间训练编码器
6. AE在训练时容易退化到一致映射(identity mapping)，使用GAN判别器进行规范化时，会使AE突出判别器所依赖的特征空间，AE解码器对AE编码器进行了"泛化"，其保证了AE编码器能生成“丰富”的信息，只有丰富信息才能更好的描述原始数据，然后通过判别器间接的“传递”到生成器

#### 11. CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms
1. 算法如何生成具有美学的艺术品，是风格转换类算法核心问题，也是最具有挑战的部分
2. 相比其它有人参与的艺术品生成算法，本论的目标是调研在创造过程中不需人干预的能够创造新的艺术品的计算机系统，
3. Martindale对创造性艺术品产生过程的解释: 创造性的艺术家通过增加艺术品的激励潜力(arousal potential)来对抗常规潜力，但是激励潜力应该努力减少观察者的负面反应；艺术家在艺术风格使用其它的方式时会增加艺术品的激励潜力，从而产生了突破性的艺术风格
4. D. E. Berlyne 认为心理学上的美学跟激励很相关，跟美学相关的激励主要有，novelty, surpris- ingness, complexity, ambiguity, and puzzlingness
5. 系统通过观察艺术品并学习风格，并通过增加潜在激励潜力提升创新性而让其不拘于原始风格，新颖但并不过于新颖
6. 一些法方通过进化算法对创新性进行探索，其基本过程是算法迭代的产生候选集，使用合适的函数进行评估，然后进行调整并进行下一轮迭代
7. GAN网络可能学习到图像中的特征，但是我们认为其在创造性方面存在局限性，修改GAN的目标使其尽量与原始风格相异并且符合艺术品本身的分布情况
8. art-generating agent配有记忆模块，记录其生产过的结果，这样可以持续生成不一样的结果，产生激励潜力的方式有多种，本论文主要关注于产生风格模糊和新异风格
9. 模糊主要有两种方式，设计(故意设计的)和本质(本质就是不可解析的)
10. CAN的生成器目标有三个：a生成新颖的作品，b但不要过于新颖，c生成的作品应该增加风格模糊
11. 判别器需要反馈两个信息给生成器：a生成的是否是艺术品，b是否是已经建立过的风格。这两个信号作用相反，最终使得生成器生成创造新的作品
12. 三个标准：a DCGAN的
13. 通过人来进行评估，让能来辨别作品是人画的，还是机器生成的
14. 实验一：选取Abstract Expressionist, Art Basel, CAN, DCGAN(64x64)等数据集，让人评估，回答两个问题：Q1是人还是机器生成，Q2是否喜欢(1-5)，结果显示CAN在Q1优于DCGAN，接近真实作品，在Q2上甚至达到了最优
15. 实验二：回答一系列问题，包括是否喜欢、是否novelty、是否surprising、是否ambiguity、是否complexity、是否由电脑生成等，结果与实验一基本一致
16. 实验三：目标是确定机器生成的作品能否视为艺术品，考察如下问题：Intentionality, Visual Structure, Communication, Inspiration, 结果让人震惊，CAN的结果比人的还好
17. 实验四：比较style classification CAN和style ambiguity CAN，主要是测试创新性，考察两个问题：哪个更具有创新性、哪个更具有美学特征；结果显示，style ambiguity CAN优于style classification CAN

#### 12. Towards Principled Methods for Training Generative Adversarial Networks
1. 没有完整的理论解析GAN的训练过程，就算有，也都是一些启发式的并且对修改很敏感，本文就是针对这类问题提出解决方案
2. GAN与VAE的主要区别在于G是如何训练的
3. 传统的GAN依赖于最大似然，可以等价于最小化Kullback-Leibler距离，但是在KL中存在明显的非对称，对于Pr(x) > Pg(x)时，特别是Pg(x)覆盖的区域，cost会非常的大；而在于Pr(x) < Pg(x)，对于错误的结果，cost会非常的小
4. 使用Jensen-shannon divergence距离可以有更好的结果
5. 如果Z的维数小于X的维数，由lemma1可以知道，g(Z)在X空间下有会一系列的0
6. 如果Pr和Pg的支撑集是独立且集中的，那么一定存在最优的判别器
7. Lemma2两个流形基本不会perfect align，那由于Lemma3由他们的交集的维数一定严格小于他们中任何一个的维数
8. 通Theorem2.1和2.2可以知道，一个完美的判别器总是存在，这样BP就不能学到任何东西了
9. logD loss function，不到出现梯度消失，但会出现不稳定的训练
10. 当M和P都是封闭时，噪声会使M和P几乎重合

#### 13. Wasserstein GAN
1. 

#### 14. Perceptual Adversarial Networks for Image-to-Image Transformation
1. PAN提出一种图片到图片之间映射的普适性框架，PAN由两个前向CNN组成，即：图片变换网络T、图片判别网络D，通过结合对抗损失和新的感知对抗损失，交替训练两个网络，到达理想的结果
2. 传统使用L1或L2距离，存正一些不可避免的问题，如模糊、缺乏感观上的信息
3. 一些论文，通过训练好的分类网络，提取高层次的特征，通过高层次信息来训练变换网络，除此之外，通地paerceptual损失来压制感观上变形
4. 不断的训练网络，可以从不同的角度抓取到不同信息

#### 15. Softmax GAN
1. 由于对判别器和生成器的训练不平衡，导致梯度消失，本文提出使用softmax cross-entropy 代替logistic loss，除非生成结果跟目标完全符合，softmax的损失不会为0
2. Least Square GAN, WGAN, Loss-Sensitive GAN这些GAN网络，都是基于无梯度消失问题的目标函数
3. Importance Sampling和NCE都是在生成模型下使用分类型进行推进，但是在NCE中使用logistic loss来区分真实数据和噪声数据，在Importance Sampling中，使用softmax cross-entropy loss进行多分类区分

#### 16. Variational Approaches for Auto-Encoding Generative Adversarial Networks
1. 使用合成的最大似然代替不可解的个似然函数、使用潜在分布代替未知的后验分布
2. AE-GAN主要可以分成三个类别：a使用ae做为判别器，b使用denoising auto-encoder做为生成器的辅助损失，c或者两者结合
3. VAE常常产生模糊的照片，但是没有GAN的模型塌缩的问题
4. 一种克服不可解的marginal likelihood方法是不去计算，通过间接的方法进行取得j5. Density Ratio Trick，假设来自数据和生成样本的可能性一样，从需跳过对p(y=1)和p(y=0)的概率计算
6. Variational Inference: 使用variational Inference的方式进行下界逼近，使用Auto-encoder
7. Synthetic Likelihoods: 使用合成的似然函数
8. VAE-GAN使用特征区分空间上的重构度量上，使用对抗损失代替似然

#### 17. Stacked Generative Adversarial Networks
1. bottom-up模式一般都是专注于抽取有用的表征，而对数据的分布无能为力
2. 引入representation discriminators用于使得SGAN的中间表示保持在DNN的流形上
3. 除了adversarial loss，还引入了conditional loss用于使得生成网络依赖于上层输入，引入novel entropy loss使得生成样本足够分散
4. 相对于使用pre-train或者perceptual loss的方法，SGAN在中间生成表示的loss，而非只关注于最后的loss
5. 由于high-level features存在不变性，在构建时其对底层也会存在不确定性，导致模糊的结果
6. 加入了conditional loss会导致conditional model collapse的问题
