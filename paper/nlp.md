## NLP related paper

### 1. Explaining and Generalizing Skip-Gram through Exponential Family Principal Component Analysis
1. Offer a new interpretation of skip-gram based on exponential family PCA---A FORM OF MATRIX FACTORIZATION.
2. Natrul language text, however, contains richer structure than simple context-word pairs. We embed n-tuples rather than pairs, allowing us to escape the bag-of-words assumption and encode richer linguistic structures.
3. 

### [2. GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
1. 使用全局非0 word-word cooccurrence matrix进行计算向量信息
2. LSA 丢失了文档词序
3. 滑动窗口等方法没有利用全局信息
4. 最大拟合词之间向量积和统计的共同出现信息
5. 相对于CBOW, n skip-gram交果要提升很多

### 3. Distributed Representations of Words and Phrases and their Compositionality(wor2vec)
1. Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships.
2. Hierarchical Softmax
3. Negative Sampling
4. Subsampling of Frequent Words
5. Learning Phrase

### 4. A Neural Attention Model for Abstractive Sentence Summarization
1. Related to neural network language models(NNLM)
2. Related to RNN
3. Attention-based model
4. 使用正文和标题做为摘要对
5. As a next step we would like to further improve the grammaticality of the sum- maries in a data-driven way, as well as scale this system to generate paragraph-level summaries.

### 5. Learning Text Similarity with Siamese Recurrent Networks
1. 使用双向LSTM siamese architecture，学习把变长的string映身
2. Siamese network 是一种非线性学习相似信息的架构
3. Siamese network 直接从相似性以及非相似性中学习不变量以及选择句子的表示
4. Siamese network 和 triplet network 也用于学习图片的相似性
5. 先计算句向量，再进行cosin相似度比较

### 6. Siamese Recurrent Architectures for Learning Sentence Similarity
1. bag-of-words/tf-idf 被其基本项限定
2. 在相似度判断中使用L2可能导致不必要的plateaus

### 7. Neural Summarization by Extracting Sentences and Words
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

### 8. Enriching Word Vectors with Subword Information(fb fasttext)
1. 形态语气词常常被忽略，但是这对于有些特定的文章会产生很多不便
2. 基于skip-gram model，每个字符表示成vector，每个词是字符的vector的和
3. 大多数现有模型把词做为一个单独的向量，而忽略了词本身结构之间的关系，这对于一些语言是十分不利的

### 9. TextRank: Bringing Order into Texts
1. Graph-based 排名算法，是一种基于全局信息对结点进行递归的排序算法。
2. 正如Graph-based 排名算法，投票越高的词其重要程度也越高
3. 使用无向带权图，定义不同的结点之前的关系
4. keyword extraction抽取文中最重要的词
5. 使用co-occurrence定义词之间的关系
6. sentence extraction抽取文中最重要的句子，使用similary来定义句子之间的相似度
7. 进行摘要

### 10. Component-Enhanced Chinese Character Embeddings
1. 基于汉字的边旁部首进行词向量提取

### 11. Character-level Convolutional Networks for Text Classification∗
1. 文本分类主要包括文本feature抽取，和分类器设计
2. 学习时，不需要语义和语法上的先验知识
3. 使用pooling可以最高计算的尝试为6 conv layer and 3 fully-connected layer
4. 使用定长的数据
5. 使用data augmentation 可以提高深度模型的范化结果，但是在文本处理中，需要考虑到其顺序，因此，采用近义词典，对词进行替换，可能得到比较好的结果。
6. Bag-of-means is misuse of word2vec

### 12. FASTTEXT.ZIP: COMPRESSING TEXT CLASSIFICATION MODELS
1. 丢失可接受的精度来获取更快的速度
2. 使用压缩，去除不重要的词
3. LSH，使用binarization strategy 进行cosin相似度搜索
4. proposed approach 提到对于不对称信息分类的一个好的方法
5. 使用PQ进行搜索，k和b的选取，使用搜索可以非常快
6. 选取一些重要的feature，并保证每个文档都被覆盖

### 13. Distributed Representations of Sentences and Documents
1. 提供一种定长向量表示句子的方法
2. 连接多个向量，求得句子向量，并由此求句子之后的单词
3. inspired使用每个词周围的词向的拼接或者均值做为当前向量的值
4. 使用huffman tree进行hierarchical softmax
5. 把句子的向量也当做词语分类的一部分
6. 保持了句子的顺序信息

### 14. A Deep Reinforced Model for Abstractive Summarization
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

### 15. Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts 2014
1. Character to Sentence Convolutional Neural Network(CharSCNN) 使用两个CNN去抽取任何大小的字符和句子的特征
2. 从character-level到sentence-level提取特征
3. Word-Level Embedding: extract syntactic and semantic information
4. Character-Level Embedding: morphological and shape information
5. 句子级别的有两个问题：句子存在不同的长度；句子的重点会出现在不同的位置

### 16. Incremental Skip-gram Model with Negative Sampling
1. 现在的词嵌算法，包括SGNG是multi-pass algorithms，因些不能进行增量模型更新
2. 分析证明，在无限数据集的情况下，increament SGNG算法效果与origin SGNG算法性能接近
3. 自适应算法更适用于增量试训练，因为数据量之前是不知道的，而且可能一直不断的增加
4. 记录TOP K个经常出现的词，这些词是一个在这K个数据中是动态出现的

### 17. Learning to Identify Ambiguous and Misleading News Headlines
1. 准确的定义问题，以及对模糊和误导进行分别对待，使用class sequential rules提取信息来处理模糊定义，使用标题和文本结合来处理误导倾向的标题
2. 使用co-training, semi-supervised method等方法来训练模型
3. 新闻标题可以分为三类：全名合理的、模糊不清的、以及误导的
4. 对于误导性标题，提取独立于正文的特征、依赖于正文的特征，并使用co-training进行训练

### 18. A Deep Network with Visual Text Composition Behavior
1. 一种新型网络，可以层级表句子的组成，还能显示低层如何关注于单个单词的
2.  Attention Gated Transformation (AGT)，每一层通过获取更多的词和句子，新的信息和之前的信息结合，产生新的表示
3. Gate由Attention控制，但是每层的结果会被上层影响
4. 分析attention和gate的分布，可以发现网络在合成词和句子的过程

### 19. Efficient Vector Representation for Documents through Corruption
1. Doc2VecC使用简单的平均词向量，并加上正则项来提取文档语义。
2. 正则项主要作用为：提升稀有或信息量大的词，并使没有区分意义的词接近0
3. Doc2VecC产生比Word2Vec更有意义的词向量
4. Doc2VecC在训练文档表示过程中同时训练词向量表示
5. Doc2VecC在训练过程中随机移除文档中的词，从而加快训练速度
6. 训练复杂度只随词的增长而增长，与文档多少无关
7. 使用Taylor Expansion对损失函数进行逼近
8. 可能在长文本处理上，效果更好

### 20. A Semantics-Based Measure of Emoji Similarity
1. 通过词嵌模型分析emoji的语相似度
2. EmoSim508数据库
3. 主要的用处：1. 检索、2. 情感分析、3. 手机键盘分析
4. deprecated

### 21. High-risk learning: acquiring new word vectors from tiny data
1. 从已有表示上学习新的词嵌表示
2. 需要从小数据中获取词向量的主要原因是：a. 一些词语非常不常见，b. 快速取得新的词的词意
3. 模拟人遇到生词的过程，从之前的词语中能个推断出当前词语的意义
4. 通过从词典中取出新词的最近K个词，来测试训练效果
5. 初始为context的和、高Learning rate、large window size

### 22. Challenges in Data-to-Document Generation
1. 随着输入的增长，输出结果的误差情况越来越严重
2. 从数据表中生成句子的主要特性有：a. 很容易生成与数据集对应的摘要，b. 摘要主要专注于覆盖数据库中的信息
3. 神经网络系统能够产生较流利的输出，而且在词级的匹配上效果也很不错，但是其内容选择和长文本处理上有所欠缺
4. 数据集提供record和描述
5. 三个标准: Content Selection(CS), Relation Generation(RG), Content Ordering(CO)

### 23. On the State of the Art of Evaluation in Neural Language Models
1. 本文揭示了一些常见的
2. Capacity and Trainability in Recurrent Neural Networks提供了一些关于rnn网络能力跟参数个数之间的关系

### 24. Learning to Ask: Neural Question Generation for Reading Comprehension
1. 介绍一基于attention的学习模型，并且调研了基于句子和段落的效果
2. 生成好的问题是一个非常难的事情，其需要抽象的生成词语，并且要求不与原文完全一样
3. 总的来说，基于规则的方法，是利用词语的语法角色，而非语义角色
4. 目前没有端到端的通过阅读生成问题的方法，也没有seq2seq的序列生成方法
5. 模弄的两个变种，只对句子进行编码的编码器，和同时对句子和段落编码的编码器
6. 解码器隐层的初始化方式，区分了基础模型和结合了段落信息的模型
7. 段落级别模型，由句子编码和段落编码拼接的向量，输入解码器解码得到问题
8. 注意力模型也分别被用在这两个模型上，句子注意力模型，只在两个模型上使用，段落级模型只在段落级模型上使用
9. 在模型训练好了之后，使用beam search得到结果
10. 输入可能存在一些稀有词，会输入为UNK，我们把其替换成生成步骤attention值最高的词
11. 在SQuAD数据集中试验
12. 句子包涵答案做为输入问题的输入句子，在答案存在多个句子时，拼接相应的句子，并句子和问题有至少一个非停词重合才算一个句子－－问题对
13. 在OpenNMT上进行训练
14. 自动评判标准：naturalness, difficulty
15. 人类评判结果生成的比人类写的还高

### 25. Effective Dimensionality Reduction for Word Embeddings
1. 对训练好的词向量，进行再进一步的处理，如降维，以适应低配设备
2. 首先对词向量进行中心化，然后使用pca比较
3. 由于向量中，信息可能被一些主导，因此除去主导信息，可以提升词向量质量
4. 发现在ppac算法之后，主导域重现，再次使用ppac有很大的必要
5. 在PPA一文中发现，词向量(包括Glove, word2vec)均值向量偏大，而且大部分能量都集中在非常小的子空间中，如果移除高能子空间的影响，可以提升词向量质量。

### 26. Long-Short Range Context Neural Networks for Language Modeling
1. 语言模型主要是为了抓取语料的统计和结构特征，短范围依赖一般用于提取语言语法特征，长距离依赖一般用于语言语义提取
2. 本文提出一种多尺度架构，本方法分别处理长和短上下文信息，并且动态的结合完成相应的任务
3. 使用Long-Short Range Context(LSRC)网络实现，其同时使用局部(short)和全局(long)上下文，并使用两个隐层表示
4. 一般来说n-gram模型不用于抓取长距离依赖，本文定义基于距离d的关联分函数，Pd(w1, w2)/(P(w1)xP(w2))，如果大于1，证明w1, w2在距离为d情况下，存在关联，通过实验，发现关联是常见的，而且跨跃的距离可以很远
5. LSTM在长距离依赖上比一般RNN表现好不少，但是没有明确指出长/短上下文表示，而是单独用一个状态进行表示
6. LSRC模型核心在于引入Hl和Hg分别抓取局部和全局信息
7. 隐变最的时序相关变化性况，可以证明LSTM, RNN, LSRC等模型在对全局和局部信息抓取的性能

### 27. A Study on Neural Network Language Modeling
1. 本文对神经网络语言模式(NNLM)进行深入研究，包括重要性采样、分类、缓存、双向循环神经网络，分别从架构和知识表示两方面阐述
2. 对于知识表示来说，NNLM一般来说是近似表示词序列在给定词料中的的统计分布，而非语言相关的知识，或者在自然语言中通过这些词序表达的信息
3. Basic NNLM: NNLM是一种统计语言模型，也可以称之为神经概率模型或神经统计模型，主要有:FNNLM, RNNLM, LSTM-RNNLM
4. Feed-forward Neural Network Language Model(FNNLM): FNNLM没有很好的方法去表示，因此一般都截取了最近的n个词用做输入，使用当前词用做输出
5. NNLM一般使用Perplexity衡量性能，其表示用于编码测试数据所要的比特位数，其值越少，代表模型越接最真实的语言情况
6. Recurrent Neural Network Language Model(RNNLM): RNNLM不仅关心输入空间，还关心内在状态空间，内在状态空间使表示序列之间的依赖关系
7. Long Short Term Memor RNNLM(LSTM-RNNLM): 用于解决长距离依赖和梯度消失等问题
8. Importance Sampling: 核心思想是使用采样的方法，近似log-likelihood gradient(可以视为两部分，目标值的正向推动，非目标真的反向推进)；为了防止发散，采样个数需随着训练进行而提升；Importance Sampling可以用于n-gram之类的模型，但不能直接用于RNNLM, LSTM-RNNLM
9. Word Class: 每个词被当成一个单独的类，在预测词时，变成了，由历史预测词的分类，再由分类和历史预测词。
10. hierarchical neural network language model: 能够加速训练，k/logk, k为所有的词数。
11. 此外，分布式词表示可以用于表示词的相似性，但是层级表示会使得这种特征减弱，层数越线，效果越差
12. Caching: 缓存语言模型是基于近期历史会有更大可能再次出现这一假设，在这一模型中，条件概率是标准模型和缓存插值；另一种是基于词类的缓存策略；在RNNLM中，通过记录输出和状态，用于未来同样的上下文环境预测
13. Bidirectional Recurrent Neural Network: 通过反转输入顺序实现，一个解释是减少了因为跨跃长度而带来的损失，本文的解析是词序列从统计角度看可能更依赖于后续文本，而非之前的内容; 通过实验，可以发现正向和反向几乎可以获取等量的信息; BiRNN可以在翻译和语音识别等问题上实现更好的效果，但其对逐词处理场景并不适用。
14. Model Architecture: 许多模型都是逐字进行处理，但是这与人在进行写作和说话的情况并不一样，一般来说，人都提前想好了要说的，或者想好了很大一部分，所以说逐字或者其它的顺序可能是不合理的
15. Knowledge Representation: 一个常见的说法是，NNLM从语料中学到了词序列的概率分布，严格来说应该是某一语料中的语序概率分布；总的来说，NNLM的知识表示是词序列在特定训练语料上的分布式表示，无论是语言本身，如语法，还是语言涉及的知识，都不是NNLM能够得到的；NNLM不能动态的从新的数据集中学习知识
16. Future: (GRU)RNNLM, dropout strategy, character level neural network language model；自然语言是人创造的，而语言知识是语言出现很久才创造的；语言知识涉及的证确词序，在真实场景中常常被误用；在现实生活中，语言只是声间或符号与抽象或真实物体的连接；因此，由于声音和符号的特片可能比自然语言更容易，可以通过找到声音和符号或物体之间的关系来处理自然语言。

### 28. A Primer on Neural Network Models for Natural Language Processing
1. nlp从sparse&linear的模型转换到dese&non-linear模型
2. 神经网络的非线性使其容易与word-embedding相结合
3. Straight-forward applications of a feed-forward network as a classifier replacement (usually coupled with the use of pre-trained word vectors) provide benefits also for CCG supertagging, dialog state tracking, pre-ordering for statistical machine translation and language modeling.
4. Convolutional and pooling architecture show promising results on many tasks, including document classification, short-text categorization, sentiment classification, relation type classification between entities, event detection, paraphrase identification, semantic role labeling, question answering, predicting box-office revenues of movies based on critic reviews, modeling text interestingness, and modeling the relation between character-sequences and part-of-speech tags.
5. Recurrent models have been shown to produce very strong results for language modeling; as well as for sequence tagging, machine translation, dependency parsing, sentiment analysis, noisy text normalization, dialog state tracking, response generation, and modeling the relation between character sequences and part-of-speech tags.
6. Recursive models were shown to produce state-of-the-art or near state-of-the-art results for constituency and dependency parse re-ranking, discourse parsing, semantic relation classification, political ideology detection based on parse trees, sentiment classification, target-dependent sentiment classification and question answering.
7. One Hot VS Dense; CBOW; Feature Combinations; Dimensionality; Vector Sharing; Network's Output
8. 非线性激励函数是nn能表示复杂信息的关键
9. HINGE: 无论是binary还是multiclass都是线性输出，HINGE LOSS在区分边界很好用，但是对成员概率判断效果有限
10. loss function: HINGE(binary, multicalss), log loss, categorical cross-entropy loss, ranking loss
11. 非监督的词向量训练方法是：相似的词有相似的向量
12. 窗口的大小对词向量的影响：大窗口着重于主题相似；小窗口着重于语法和功能相似
13. 词向量与语法信息十分相关
14. SGD & Computation graph
15. Initialization: xavier initialization, zero-mean Gaussian for ReLU
16. Vanishing and Exploding gradients: shallower network, step-wise training, performing batch-normalization, LSTM, GRU, clipping the gradients if their norm exceeds a given threshold
17. 如果网络不好使，要关注网络中saturation and dead neurons; saturation: 一般是输入值过大; dead: 梯度过大
18. Shuffling & Learning Rate & Minibatches
19. Regularization & dropout
20. Model Cascading & Multi-task Learning
21. NLP的输出一般不是分类标签或者类别分布，而是输出序列，树，或图
22. Greedy Structured Prediction & Search Based Structured Prediction & Probabilistic Objective (CRF) & Reranking & MEMM and Hybrid Approaches
23. Convolutional Layers: ordered sets of items; CBOW完全取忽略了词序; Dynamic, Hierarchical and k-max Pooling;
24. Recurrent Neural Networks – Modeling Sequences and Stacks: backpropagation through time; Acceptor 使用最后的结果作为输出; Encoder最后的信息用作前面所有信息的编码; Transducer使用每一步的输出; Encoder - Decoder; Multi-layer (Stacked) RNNs; Bidirectional RNNs (biRNN); RNNs for Representing Stacks; 
25. Concrete RNN Architectures: Simple RNN; LSTM; 
26. Modeling Trees – Recursive Neural Networks

### 29. A Convolutional Neural Network for Modelling Sentences
1. Dynamic Convolutional Neural Network(DCNN)用于对句子语义建模
2. 网络使用动态k-Max Pooling
3. 网络可能处理不同的输入长的句子，并产生能够抓住长短距离的关系的特征图
4. 算法的核心在于如何从句子的n-gram或词中抽取特征
5. a. k-max pooling句子，取得top-k maximun，b. k可以动态选取，可以视为其它特征的函数
6. one dimensional filter for each n-gram => k-max pooling and no-linearity => feature map of the sentence
7. 高层的small filter可以抓住句子中远距离词之间语法或语义关系
8. One-dimensional convolution的基本观点是：使用filter与每个n-gram进行点乘得到另一个序列
9. wide convolution 相对于narrow convolution有一定优势
10. Max-TDNN pros: 1. 对句子词序敏感，且不依赖外在结构；2. 每个词都有同等重要性；cons: 1. 只能考虑m范围的词语; 2. 当扩大filter或增加层数时，对句子的最小长度有基本要求
11. Max-pooling cons: 1. 不能区分相应的feature是否多次出现; 2. 每次pooling过多的删减信息
12. DCNN生成的树比语法树更一般化，其不被一些特点的语法约束，而且可以处理长短两种语义关系

### 30. word2vec Parameter Learning Explained
1. 对word2vec相关计算原理，过程都有详细的分析介绍

### 31. Hierarchical Probabilistic Neural Network Language Model
1. 神经网络在语言模型中有很多成功的应用，其主要的优点在于在连续空间表示词(或者符号)，平滑了语言模型，这样即使在训练数据不足的情况下，也能有很好的泛化效果。
2. 相关模型速度相当慢，本文提出基于条件概率的层级分级方法来加速训练，可以使训练和识别加速到200倍
3. word classes: 根据词语使用在上下文使用的概率来把词语分成不同的分块.
4. 层级分解的基本思想是把线性预测，分解成log(V)步的决策，从而加快了计算; 另一个方法是，使用同一个模型来处理所有的决策
5. 基本思想是：不直接计算上下文与某个词的关系时，而是计算上下文预测词所属的类别，再通过类别预测词，从而加速计算
6. 每个词可以表示成一个十进制串，大二叉层级树中进行索引
7. 采用平衡树，或者huffman tree编码可以进一步加快计算速度

### 32. Bilingual Word Embeddings for Phrase-Based Machine Translation
1. 通过学习两个语言对应词的embedding，来优化机器翻译对齐和效果
2. 新的词嵌方法，在词语相似方面，显著的提升了效果
3. 跨语言的词嵌训练，可以增加跨语言的相似程度
4. 通过无监督的方法，来训练双语词嵌，主要是方法是拉开文档中正确的词和随机词的评分间隔

### 33. Training RNNs as Fast as CNNs
1. 通过简化状态计算得到更具有并行性的RNN实现


### 34. StarSpace: Embed All The Things!
1. 一个通用的神经网络嵌入模型，可以处理: labeling tasks such as text classification, ranking tasks such as in- formation retrieval/web search, collaborative filtering-based or content-based recommendation, embedding of multi- relational graphs, and learning word, sentence or document level embeddings.
2. 嵌入由不同离散特征组成的实体，并计算他们之间的相似性。
3. StarSpace模型由不同的learning entities组成，每个entity由一组离散的features组成，如词语则可以视为n-gram模型。
4. 在StartSpace模型中，任意的实体之间可以比较，即使他们是不同的种类。
5. 选对feature进行建模，每个Feature由一个d维向量构成k
6. Multiclass Classification: (a, b), (a, b-)，用采样技术生成
7. Multilabel Classification: 每个文档有多个标签，从中采成(a, b)对进行训练
8. Collaborative Filtering-based Recommendation: 每个用户用其喜欢的物品表示，一个唯一的值表示用用户，用户与其喜欢的物品组成正例，其它的组成负例
9. Collaborative Filtering-based Recommendation with out-of-sample user extension: 不用ID表示用户，而是且其喜欢的物品集合(不包括其中一个)，与这一个组成正例
10. Content-based Recommendation: 用户由其喜欢的物品描述，物品由其特征描述
11. Multi-Relational Knowledge Graphs (e.g. Link Predic- tion)
12. Information Retrieval (e.g. Document Search) and Document Embeddings: 监督与非监督两种方式
13. Learning Word Embeddings: word2vec
14. Learning Sentence Embeddings: 使用来自同一文章的句子为正例

### 35. Think Globally, Embed Locally — Locally Linear Meta-embedding of Words
1. 通过结合已有词嵌产生更加精确完善的meta-embedding
2. 提出一种无监督算法，使用预训词向量做为输入，生产更加精确的meta-embedding
3. 已经提出的词向量拼接，可以看成本文的一种特殊实例
4. 面临的问题：训练语料不相同; 输入词向量维度不相同; 不同的词向量中，词的邻近词有很大的不同
5. 提出locally-linear meta-embedding学习方法，a) 只需要在词表中的词，b) 可以meta-embed不同长度的源词嵌，c) 对不同词嵌的邻近变化敏感
6. 算法主要分为两步：a) recosntuction step: 对每个源词嵌进行细性结合每个词的最邻近词; b) projection step: 计算meta-embedding
7. 本方法解决了源词嵌不对齐，以及词语缺失等问题
8. 使用BallTree algorithm加速搜索
9. 不同的词嵌来源，其优化目标也不一样，这样邻近词表就会进行互补，从而达到更优的效果
10. 下一步方向，跨语言meta-embedding实现

### 36. BLEU: a Method for Automatic Evaluation of Machine Translation
1. 提出一种自动机器翻译度量方法，具有语言独立、类似人工衡量、并且高效
2. 翻译的三个方面，adequacy, fidelity, fluency
3. BLUE最基本准则是，对比并统计候选结果与真实结果应用n-gram后匹配的个数
4. 1-gram用于保证adequacy, 更长的n-gram增加流畅性
5. 在单句上不同的人衡量结果有所不同，BLUE一般用于平均性能
6. 对于不存在的词，以及出现过多的词进行惩罚
7. 一个词对，应该只依赖一个参考，而非多个
8. brevity penality: 关注长度、词语、词序

### 37. Efficient Estimation of Word Representations in Vector Space
1. 使用Huffman binary tree做为层级softmax
2. CBOW
3. Skip-gram
4. skip-gram在semantic和syntactic要优于CBOW

### 38. Linguistic Regularities in Continuous Space Word Representations
1. 使用输入层权重隐性表示词的向量，可以很好的抓住语法和语义特征，并且每一种特征都具有特征空间偏移特性
2. 训练一个网络，一方面可以得到模型本身，另一方面，可以得到词的表示
3. 使用网络训练，由上文预测下文
4. 提供一个offset衡量分布式表示的方法
5. 所有结果，以及附加品，都是无监督学习取得
6. 基于cosine distance的偏移测量
7. 语义和语法测试集

### 39. WORD TRANSLATION WITHOUT PARALLEL DATA
1. 不需要平行语料，只需要单一语言语料
2. 训练分为两步，a. 对抗训练，把两种语言映射到目标空间，b. 抽取合成词典，转化成the closed-form Procrustes solution
3. 提出一种与映射质量高相关的非监督筛选标准
4. 给出12种语言的词典，并给出相应的监督与非监督训练对
5. 使用CSLS生成词典
6. 由于低频词效果有限，只训练top 50000词
7. 保证W矩阵接近正交，可以增加效果
8. 计算翻译的平均cosin距离，用作验证标准

### 40. Hierarchical Probabilistic Neural Network Language Model
1. 层级结构

### 41. DisSent: Sentence Representation Learning from Explicit Discourse Relations
1. 使用'because but, although等词来标识句子之间的关系。
2. 使用双向lstm encode句子，然后进行时域maxpooling，然后向量相加
3. 使用依存分析去掉一些discourse使用

### 42. Learning Distributed Representations of Sentences from Unlabelled Data
1. 提出一种新的句子或短语训练目标- Sequential Denoising Autoencoders(SDAEs)和FastSent
2. 不使用botton up策略，pharse或sentence表示是在词向量的数学运算的基础上
3. k

### 43. Advances in Pre-Training Distributed Word Representations
1. 提出一种新的词向量预训方法
2. 几种很少一起使用的策略：the position dependent features，the phrase rep- resentations，the use of subword information
3. 原始word2vec只是简单的平均窗口中的词，并没有考虑位置信息
4. 原始cbow只是unigrams，没有考虑到词序
5. 标准的词向量忽略了词的内部结构

### 44. Zero-Shot Relation Extraction via Reading Comprehension
1. 一般的方法不能抽取未提前说明或没在训练中见到过的实体
2. 在数据和模型都提出了新方法，1) 使用距离监督来处理巨量的关系，2) 使用众包的方法来收集和验证每一个关系的问题
3. 在解决阅读理角问题上，与解决关系抽取填空问题存在一定的相似性
4. 问题的难点在于把关系转化成问题，而非处理每个实体的答案，这样在标时也从实体标转化成为关系标注，提升了效率
5. 第一个schema level基于众包的qa数据集

### 45. Neural Relation Extraction with Selective Attention over Instances
1. 远程监督关系不可避免的伴随标注错误问题
2. 提出句子级基于注意力机制的关系抽取方法
3. 1) 相比于现有的神经网络关系抽取，本模型可以充分利用实体间的信息，2) 使用选择性注意力来降低noise的影响，3) 实验结果显示有效
4. 与别的模型不同的是，提出在多个实体上的句子级注意力机制，可以利用所有的信息
5. 模型分为两个阶段：1) 学习句子的分布式表示，2) 使用句子级注意力机制选取表达关系的句子

### 46. DIALOGUE LEARNING WITH HUMAN-IN-THE-LOOP
1. 一般的对话机器人训练着重于在固定数据集上训练，本文使用强化学习基于在线反馈训练在线模型。
2. 本文考虑两种反馈：明确的奖励，文本回应
3. 本文的模型policy可以视为在数据集上的多次迭代
4. 数据测试分为两步，1) 模拟环境，2) Mechanical Turk平台
5. 本文只考虑，Task 6，当BOT回答正确时，Teacher给出正面回答，当机器人回答错误时，Teacher给出文本答案。
6. 底层网络使用MemN2N
7. 1) 转化文本为向量表示，2) 转化记忆为向量表示，3) 选出与文本相关的记忆单元，4) 记忆可以进行多次查询，5) 使用最后查询向量表示与候选者进行softmax，得到相应概率
8. policy: MemN2N, state: 聊天记录，action: MemN2N结果，reword: 对(1),错(0)，
9. 使用batch size来确定模型更新时间

### 47. 360° Stance Detection
1. 一个使用自然语言检测言论风向的应用

### 48. Investigating Capsule Networks with Dynamic Routing for Text Classification
1. 使用了三个方法来减轻capsules的扰动，并在6个任务中4个达到了较好的结果
2. 使用capsules的向量输出，来代替CNN的标量输出，用于保证局部的词序信息。

### 49. THE UNREASONABLE EFFECTIVENESS OF THE FORGET GATE
1. provided another network just use the forget gate
2. half parameter and two-third of the element-wise multiplications.
3. the key is the parameter initialization

### 50. Scalable attribute-aware network embedding with localily
1. 网络词向量可以方便联合学习拓扑和属性
2. linearity and scalable

### 51. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
1. 本文发现，每简单的卷积网络都可以超过权威的循环神经网络。
2. 有许多论文对各种循环神经网络进行过验证。
3. 本文结合Causal Convolutions、Dilated Convolutions、Residual Connections得到TCN
4. TCN的主要优点有：并行、灵活的接收域、稳定的梯度、训练占用内存少、可变长输入。
5. 本文在The adding problem、Sequential MNIST and P-MNIST、Copy memory、JSB Chorales and Nottingham、PennTreebank、Wikitext-103、LAMBADA、text8等多个数据集上进行试验，与常见的LSTM、GRU等网络进行对比，在结果和性能上TCN都取得了相当不错的优势。

### 52. Attention is all you need
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Using dk to pushing the softmax function into regions where is has extremely small gradients.
4. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
5. Positional Encoding: position embeding and sinusoidal

### 53. OpenSeq2Seq: extensible toolkit for distributed and mixed precision training of sequence-to-sequence models
1. seq2seq, and fully supporting ditributed and mixed-precision training
2. any encoder can combined with any decoder

### 54. Deep contextualized word representations
1. ELMo representations are deep, in the sense that they are a function of all of the internal layers of the biLM
2. we show that the higher-level LSTM states capture context-dependent aspects of word meaning,  while lower level states model aspects of syntax
3. Our approach also benefits from subword units through the use of character convolutions, and we seamlessly incorporate multi-sense information into downstream tasks without explicitly training to predict predefined sense classes.

### 52. Paper Abstract Writing through Editing Mechanism

### node2vec: Scalable feature learning for networks
### Line: Large-scale information network embedding. 
### Deepwalk:Onlinelearning of social representations.
### Nonlinear dimensionality reduction by locally linear embedding(12142)
### Aglobalgeometric framework for nonlinear dimensionality reduction(10000)
### 50. Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering


### 50. LSTM: A Search Space Odyssey


### 49. Linguistically regularized lstms for sentiment classification

### 44. Recent Advances in Recurrent Neural Networks
1. 

### 22. Recurrent neural network based language model

