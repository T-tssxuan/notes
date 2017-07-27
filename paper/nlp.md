## NLP related paper

#### 1. Explaining and Generalizing Skip-Gram through Exponential Family Principal Component Analysis
1. Offer a new interpretation of skip-gram based on exponential family PCA---A FORM OF MATRIX FACTORIZATION.
2. Natrul language text, however, contains richer structure than simple context-word pairs. We embed n-tuples rather than pairs, allowing us to escape the bag-of-words assumption and encode richer linguistic structures.
3. 

#### [2. GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
1. 使用全局非0 word-word cooccurrence matrix进行计算向量信息
2. LSA 丢失了文档词序
3. 滑动窗口等方法没有利用全局信息
4. 最大拟合词之间向量积和统计的共同出现信息
5. 相对于CBOW, n skip-gram交果要提升很多

#### 3. Distributed Representations of Words and Phrases and their Compositionality(wor2vec)
1. Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships.
2. Hierarchical Softmax
3. Negative Sampling
4. Subsampling of Frequent Words
5. Learning Phrase

#### 4. A Neural Attention Model for Abstractive Sentence Summarization
1. Related to neural network language models(NNLM)
2. Related to RNN
3. Attention-based model
4. 使用正文和标题做为摘要对
5. As a next step we would like to further improve the grammaticality of the sum- maries in a data-driven way, as well as scale this system to generate paragraph-level summaries.

#### 5. Learning Text Similarity with Siamese Recurrent Networks
1. 使用双向LSTM siamese architecture，学习把变长的string映身
2. Siamese network 是一种非线性学习相似信息的架构
3. Siamese network 直接从相似性以及非相似性中学习不变量以及选择句子的表示
4. Siamese network 和 triplet network 也用于学习图片的相似性
5. 先计算句向量，再进行cosin相似度比较

#### 6. Siamese Recurrent Architectures for Learning Sentence Similarity
1. bag-of-words/tf-idf 被其基本项限定
2. 在相似度判断中使用L2可能导致不必要的plateaus

#### 7. Neural Summarization by Extracting Sentences and Words
1. 使用encode抽到句子的表示，使用decoder进行理解
2. 使用单个句子进行训练，而非使用标题以及第一句进行抽取
3. decoder从文本中选取，而非从整个词表中抽取
4. 结果字符集是从原文档中抽取的
5. 使用规则对文章中的句子进行判断，如果包括某些词，则认为是摘要
6. 对于oov则进行，则基于词向量找到相近的句子
7. neural network-based hierarchical document reader and an attention-based hierarchical content extractor.
8. 全局信息被提取，局部信息被保留
9. 使用CNN提取句子级别的表示，使用RNN提取文档级别的表示
10. 对比于 seq2seq，本文是直接输出突出的句子

#### 8. Enriching Word Vectors with Subword Information(fb fasttext)
1. 形态语气词常常被忽略，但是这对于有些特定的文章会产生很多不便
2. 基于skip-gram model，每个字符表示成vector，每个词是字符的vector的和
3. 大多数现有模型把词做为一个单独的向量，而忽略了词本身结构之间的关系，这对于一些语言是十分不利的

#### 8. TextRank: Bringing Order into Texts
1. Graph-based 排名算法，是一种基于全局信息对结点进行递归的排序算法。
2. 正如Graph-based 排名算法，投票越高的词其重要程度也越高
3. 使用无向带权图，定义不同的结点之前的关系
4. keyword extraction抽取文中最重要的词
5. 使用co-occurrence定义词之间的关系
6. sentence extraction抽取文中最重要的句子，使用similary来定义句子之间的相似度
7. 进行摘要

#### 9. Component-Enhanced Chinese Character Embeddings
1. 基于汉字的边旁部首进行词向量提取

#### 10. Character-level Convolutional Networks for Text Classification∗
1. 文本分类主要包括文本feature抽取，和分类器设计
2. 学习时，不需要语义和语法上的先验知识
3. 使用pooling可以最高计算的尝试为6 conv layer and 3 fully-connected layer
4. 使用定长的数据
5. 使用data augmentation 可以提高深度模型的范化结果，但是在文本处理中，需要考虑到其顺序，因此，采用近义词典，对词进行替换，可能得到比较好的结果。
6. Bag-of-means is misuse of word2vec

#### 11. FASTTEXT.ZIP: COMPRESSING TEXT CLASSIFICATION MODELS
1. 丢失可接受的精度来获取更快的速度
2. 使用压缩，去除不重要的词
3. LSH，使用binarization strategy 进行cosin相似度搜索
4. proposed approach 提到对于不对称信息分类的一个好的方法
5. 使用PQ进行搜索，k和b的选取，使用搜索可以非常快
6. 选取一些重要的feature，并保证每个文档都被覆盖

#### 12. Distributed Representations of Sentences and Documents
1. 提供一种定长向量表示句子的方法
2. 连接多个向量，求得句子向量，并由此求句子之后的单词
3. inspired使用每个词周围的词向的拼接或者均值做为当前向量的值
4. 使用huffman tree进行hierarchical softmax
5. 把句子的向量也当做词语分类的一部分
6. 保持了句子的顺序信息

#### 13. A Deep Reinforced Model for Abstractive Summarization
1. 在编码器中使用intra-temporal attention记录已经输入的词的关注权重，并在解码器中使用intra-temporal把已经输出的词考虑进来
2. 结合maximum-liklihood cross-entropy loss和reinforcement learning的奖励机制来降低exposure bias
3. 使用intra-temporal attention function机制来防止在不同的步骤关注相关的部分，此机制主要包括对decoder的隐状态、已生成词和encoder输入序列
4. 使用decode查看之前decoder步骤来进一步防止重复生成相同的词
5. 使用shared weight更适合summarization，另外，使用这种方式，可以加快模型收敛
6. 通个观察发现，人生成的摘要基本不会出现同一个三元组两次
7. 使用强化学习来训练任务
8. 有监督网络在训练时，总是知道下一个结果是什么，这样可以调整，但是测试时，会导致误差被累积
9. 使用强化学习和有监督学习来完成训练，其中有监督学习提升信息抓取力度、强化学习提升可读性
10. 把标题、署名、内容使用特殊符号连接，全部小字，数字变成0，
11. 90%训练、5%validation, 5%testing
12. 首先使用maximum-likelihood训练，再进行ml + rl训练，200LSTM encode，400LSTM decode，15000词输入，5000词输出，通过选择出现频率最高的词，100维的词向量

#### 14. Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts 2014
1. Character to Sentence Convolutional Neural Network(CharSCNN) 使用两个CNN去抽取任何大小的字符和句子的特征
2. 从character-level到sentence-level提取特征
3. Word-Level Embedding: extract syntactic and semantic information
4. Character-Level Embedding: morphological and shape information
5. 句子级别的有两个问题：句子存在不同的长度；句子的重点会出现在不同的位置

#### 15. Incremental Skip-gram Model with Negative Sampling
1. 现在的词嵌算法，包括SGNG是multi-pass algorithms，因些不能进行增量模型更新
2. 分析证明，在无限数据集的情况下，increament SGNG算法效果与origin SGNG算法性能接近
3. 自适应算法更适用于增量试训练，因为数据量之前是不知道的，而且可能一直不断的增加
4. 记录TOP K个经常出现的词，这些词是一个在这K个数据中是动态出现的

#### 16. Learning to Identify Ambiguous and Misleading News Headlines
1. 准确的定义问题，以及对模糊和误导进行分别对待，使用class sequential rules提取信息来处理模糊定义，使用标题和文本结合来处理误导倾向的标题
2. 使用co-training, semi-supervised method等方法来训练模型
3. 新闻标题可以分为三类：全名合理的、模糊不清的、以及误导的
4. 对于误导性标题，提取独立于正文的特征、依赖于正文的特征，并使用co-training进行训练

#### 17. A Deep Network with Visual Text Composition Behavior
1. 一种新型网络，可以层级表句子的组成，还能显示低层如何关注于单个单词的
2.  Attention Gated Transformation (AGT)，每一层通过获取更多的词和句子，新的信息和之前的信息结合，产生新的表示
3. Gate由Attention控制，但是每层的结果会被上层影响
4. 分析attention和gate的分布，可以发现网络在合成词和句子的过程

#### 18. Efficient Vector Representation for Documents through Corruption
1. Doc2VecC使用简单的平均词向量，并加上正则项来提取文档语义。
2. 正则项主要作用为：提升稀有或信息量大的词，并使没有区分意义的词接近0
3. Doc2VecC产生比Word2Vec更有意义的词向量
4. Doc2VecC在训练文档表示过程中同时训练词向量表示
5. Doc2VecC在训练过程中随机移除文档中的词，从而加快训练速度
6. 训练复杂度只随词的增长而增长，与文档多少无关
7. 使用Taylor Expansion对损失函数进行逼近
8. 可能在长文本处理上，效果更好

#### 19. A Semantics-Based Measure of Emoji Similarity
1. 通过词嵌模型分析emoji的语相似度
2. EmoSim508数据库
3. 主要的用处：1. 检索、2. 情感分析、3. 手机键盘分析
4. deprecated

#### 20. High-risk learning: acquiring new word vectors from tiny data
1. 从已有表示上学习新的词嵌表示
2. 需要从小数据中获取词向量的主要原因是：a. 一些词语非常不常见，b. 快速取得新的词的词意
3. 模拟人遇到生词的过程，从之前的词语中能个推断出当前词语的意义
4. 通过从词典中取出新词的最近K个词，来测试训练效果
5. 初始为context的和、高Learning rate、large window size

#### 21. Challenges in Data-to-Document Generation
1. 随着输入的增长，输出结果的误差情况越来越严重
2. 从数据表中生成句子的主要特性有：a. 很容易生成与数据集对应的摘要，b. 摘要主要专注于覆盖数据库中的信息
3. 神经网络系统能够产生较流利的输出，而且在词级的匹配上效果也很不错，但是其内容选择和长文本处理上有所欠缺
4. 

#### 22. Recurrent neural network based language model
