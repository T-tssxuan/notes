## NLP related paper

### 1. Explaining and Generalizing Skip-Gram through Exponential Family Principal Component Analysis
- Offer a new interpretation of skip-gram based on exponential family PCA---A FORM OF MATRIX FACTORIZATION.
- Natrul language text, however, contains richer structure than simple context-word pairs. We embed n-tuples rather than pairs, allowing us to escape the bag-of-words assumption and encode richer linguistic structures.
- 

### [2. GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- 使用全局非0 word-word cooccurrence matrix进行计算向量信息
- LSA 丢失了文档词序
- 滑动窗口等方法没有利用全局信息
- 最大拟合词之间向量积和统计的共同出现信息
- 相对于CBOW, n skip-gram交果要提升很多

### 3. Distributed Representations of Words and Phrases and their Compositionality(wor2vec)
- Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships.
- Hierarchical Softmax
- Negative Sampling
- Subsampling of Frequent Words
- Learning Phrase

### 4. A Neural Attention Model for Abstractive Sentence Summarization
- Related to neural network language models(NNLM)
- Related to RNN
- Attention-based model
- 使用正文和标题做为摘要对
- As a next step we would like to further improve the grammaticality of the sum- maries in a data-driven way, as well as scale this system to generate paragraph-level summaries.

### 5. Learning Text Similarity with Siamese Recurrent Networks
- 使用双向LSTM siamese architecture，学习把变长的string映身
- Siamese network 是一种非线性学习相似信息的架构
- Siamese network 直接从相似性以及非相似性中学习不变量以及选择句子的表示
- Siamese network 和 triplet network 也用于学习图片的相似性
- 先计算句向量，再进行cosin相似度比较

### 6. Siamese Recurrent Architectures for Learning Sentence Similarity
- bag-of-words/tf-idf 被其基本项限定
- 在相似度判断中使用L2可能导致不必要的plateaus

### 7. Neural Summarization by Extracting Sentences and Words
- 使用encode抽到句子的表示，使用decoder进行理解
- 使用单个句子进行训练，而非使用标题以及第一句进行抽取
- decoder从文本中选取，而非从整个词表中抽取
- 结果字符集是从原文档中抽取的
- 使用规则对文章中的句子进行判断，如果包括某些词，则认为是摘要
- 对于oov则进行，则基于词向量找到相近的句子
- neural network-based hierarchical document reader and an attention-based hierarchical content extractor.
- 全局信息被提取，局部信息被保留
- 使用CNN提取句子级别的表示，使用RNN提取文档级别的表示
- 对比于 seq2seq，本文是直接输出突出的句子

### 8. Enriching Word Vectors with Subword Information(fb fasttext)
- 形态语气词常常被忽略，但是这对于有些特定的文章会产生很多不便
- 基于skip-gram model，每个字符表示成vector，每个词是字符的vector的和
- 大多数现有模型把词做为一个单独的向量，而忽略了词本身结构之间的关系，这对于一些语言是十分不利的

### 9. TextRank: Bringing Order into Texts
- Graph-based 排名算法，是一种基于全局信息对结点进行递归的排序算法。
- 正如Graph-based 排名算法，投票越高的词其重要程度也越高
- 使用无向带权图，定义不同的结点之前的关系
- keyword extraction抽取文中最重要的词
- 使用co-occurrence定义词之间的关系
- sentence extraction抽取文中最重要的句子，使用similary来定义句子之间的相似度
- 进行摘要

### 10. Component-Enhanced Chinese Character Embeddings
- 基于汉字的边旁部首进行词向量提取

### 11. Character-level Convolutional Networks for Text Classification∗
- 文本分类主要包括文本feature抽取，和分类器设计
- 学习时，不需要语义和语法上的先验知识
- 使用pooling可以最高计算的尝试为6 conv layer and 3 fully-connected layer
- 使用定长的数据
- 使用data augmentation 可以提高深度模型的范化结果，但是在文本处理中，需要考虑到其顺序，因此，采用近义词典，对词进行替换，可能得到比较好的结果。
- Bag-of-means is misuse of word2vec

### 12. FASTTEXT.ZIP: COMPRESSING TEXT CLASSIFICATION MODELS
- 丢失可接受的精度来获取更快的速度
- 使用压缩，去除不重要的词
- LSH，使用binarization strategy 进行cosin相似度搜索
- proposed approach 提到对于不对称信息分类的一个好的方法
- 使用PQ进行搜索，k和b的选取，使用搜索可以非常快
- 选取一些重要的feature，并保证每个文档都被覆盖

### 13. Distributed Representations of Sentences and Documents
- 提供一种定长向量表示句子的方法
- 连接多个向量，求得句子向量，并由此求句子之后的单词
- inspired使用每个词周围的词向的拼接或者均值做为当前向量的值
- 使用huffman tree进行hierarchical softmax
- 把句子的向量也当做词语分类的一部分
- 保持了句子的顺序信息

### 14. A Deep Reinforced Model for Abstractive Summarization
- 在编码器中使用intra-temporal attention记录已经输入的词的关注权重，并在解码器中使用intra-temporal把已经输出的词考虑进来
- 结合maximum-liklihood cross-entropy loss和reinforcement learning的奖励机制来降低exposure bias
- 使用intra-temporal attention function机制来防止在不同的步骤关注相关的部分，此机制主要包括对decoder的隐状态、已生成词和encoder输入序列
- 使用decode查看之前decoder步骤来进一步防止重复生成相同的词
- 使用shared weight更适合summarization，另外，使用这种方式，可以加快模型收敛
- 通个观察发现，人生成的摘要基本不会出现同一个三元组两次
- 使用强化学习来训练任务
- 有监督网络在训练时，总是知道下一个结果是什么，这样可以调整，但是测试时，会导致误差被累积
- 使用强化学习和有监督学习来完成训练，其中有监督学习提升信息抓取力度、强化学习提升可读性
- 把标题、署名、内容使用特殊符号连接，全部小字，数字变成0，
- 90%训练、5%validation, 5%testing
- 首先使用maximum-likelihood训练，再进行ml + rl训练，200LSTM encode，400LSTM decode，15000词输入，5000词输出，通过选择出现频率最高的词，100维的词向量

### 15. Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts 2014
- Character to Sentence Convolutional Neural Network(CharSCNN) 使用两个CNN去抽取任何大小的字符和句子的特征
- 从character-level到sentence-level提取特征
- Word-Level Embedding: extract syntactic and semantic information
- Character-Level Embedding: morphological and shape information
- 句子级别的有两个问题：句子存在不同的长度；句子的重点会出现在不同的位置

### 16. Incremental Skip-gram Model with Negative Sampling
- 现在的词嵌算法，包括SGNG是multi-pass algorithms，因些不能进行增量模型更新
- 分析证明，在无限数据集的情况下，increament SGNG算法效果与origin SGNG算法性能接近
- 自适应算法更适用于增量试训练，因为数据量之前是不知道的，而且可能一直不断的增加
- 记录TOP K个经常出现的词，这些词是一个在这K个数据中是动态出现的

### 17. Learning to Identify Ambiguous and Misleading News Headlines
- 准确的定义问题，以及对模糊和误导进行分别对待，使用class sequential rules提取信息来处理模糊定义，使用标题和文本结合来处理误导倾向的标题
- 使用co-training, semi-supervised method等方法来训练模型
- 新闻标题可以分为三类：全名合理的、模糊不清的、以及误导的
- 对于误导性标题，提取独立于正文的特征、依赖于正文的特征，并使用co-training进行训练

### 18. A Deep Network with Visual Text Composition Behavior
- 一种新型网络，可以层级表句子的组成，还能显示低层如何关注于单个单词的
-  Attention Gated Transformation (AGT)，每一层通过获取更多的词和句子，新的信息和之前的信息结合，产生新的表示
- Gate由Attention控制，但是每层的结果会被上层影响
- 分析attention和gate的分布，可以发现网络在合成词和句子的过程

### 19. Efficient Vector Representation for Documents through Corruption
- Doc2VecC使用简单的平均词向量，并加上正则项来提取文档语义。
- 正则项主要作用为：提升稀有或信息量大的词，并使没有区分意义的词接近0
- Doc2VecC产生比Word2Vec更有意义的词向量
- Doc2VecC在训练文档表示过程中同时训练词向量表示
- Doc2VecC在训练过程中随机移除文档中的词，从而加快训练速度
- 训练复杂度只随词的增长而增长，与文档多少无关
- 使用Taylor Expansion对损失函数进行逼近
- 可能在长文本处理上，效果更好

### 20. A Semantics-Based Measure of Emoji Similarity
- 通过词嵌模型分析emoji的语相似度
- EmoSim508数据库
- 主要的用处：1. 检索、2. 情感分析、3. 手机键盘分析
- deprecated

### 21. High-risk learning: acquiring new word vectors from tiny data
- 从已有表示上学习新的词嵌表示
- 需要从小数据中获取词向量的主要原因是：a. 一些词语非常不常见，b. 快速取得新的词的词意
- 模拟人遇到生词的过程，从之前的词语中能个推断出当前词语的意义
- 通过从词典中取出新词的最近K个词，来测试训练效果
- 初始为context的和、高Learning rate、large window size

### 22. Challenges in Data-to-Document Generation
- 随着输入的增长，输出结果的误差情况越来越严重
- 从数据表中生成句子的主要特性有：a. 很容易生成与数据集对应的摘要，b. 摘要主要专注于覆盖数据库中的信息
- 神经网络系统能够产生较流利的输出，而且在词级的匹配上效果也很不错，但是其内容选择和长文本处理上有所欠缺
- 数据集提供record和描述
- 三个标准: Content Selection(CS), Relation Generation(RG), Content Ordering(CO)

### 23. On the State of the Art of Evaluation in Neural Language Models
- 本文揭示了一些常见的
- Capacity and Trainability in Recurrent Neural Networks提供了一些关于rnn网络能力跟参数个数之间的关系

### 24. Learning to Ask: Neural Question Generation for Reading Comprehension
- 介绍一基于attention的学习模型，并且调研了基于句子和段落的效果
- 生成好的问题是一个非常难的事情，其需要抽象的生成词语，并且要求不与原文完全一样
- 总的来说，基于规则的方法，是利用词语的语法角色，而非语义角色
- 目前没有端到端的通过阅读生成问题的方法，也没有seq2seq的序列生成方法
- 模弄的两个变种，只对句子进行编码的编码器，和同时对句子和段落编码的编码器
- 解码器隐层的初始化方式，区分了基础模型和结合了段落信息的模型
- 段落级别模型，由句子编码和段落编码拼接的向量，输入解码器解码得到问题
- 注意力模型也分别被用在这两个模型上，句子注意力模型，只在两个模型上使用，段落级模型只在段落级模型上使用
- 在模型训练好了之后，使用beam search得到结果
- 输入可能存在一些稀有词，会输入为UNK，我们把其替换成生成步骤attention值最高的词
- 在SQuAD数据集中试验
- 句子包涵答案做为输入问题的输入句子，在答案存在多个句子时，拼接相应的句子，并句子和问题有至少一个非停词重合才算一个句子－－问题对
- 在OpenNMT上进行训练
- 自动评判标准：naturalness, difficulty
- 人类评判结果生成的比人类写的还高

### 25. Effective Dimensionality Reduction for Word Embeddings
- 对训练好的词向量，进行再进一步的处理，如降维，以适应低配设备
- 首先对词向量进行中心化，然后使用pca比较
- 由于向量中，信息可能被一些主导，因此除去主导信息，可以提升词向量质量
- 发现在ppac算法之后，主导域重现，再次使用ppac有很大的必要
- 在PPA一文中发现，词向量(包括Glove, word2vec)均值向量偏大，而且大部分能量都集中在非常小的子空间中，如果移除高能子空间的影响，可以提升词向量质量。

### 26. Long-Short Range Context Neural Networks for Language Modeling
- 语言模型主要是为了抓取语料的统计和结构特征，短范围依赖一般用于提取语言语法特征，长距离依赖一般用于语言语义提取
- 本文提出一种多尺度架构，本方法分别处理长和短上下文信息，并且动态的结合完成相应的任务
- 使用Long-Short Range Context(LSRC)网络实现，其同时使用局部(short)和全局(long)上下文，并使用两个隐层表示
- 一般来说n-gram模型不用于抓取长距离依赖，本文定义基于距离d的关联分函数，Pd(w1, w2)/(P(w1)xP(w2))，如果大于1，证明w1, w2在距离为d情况下，存在关联，通过实验，发现关联是常见的，而且跨跃的距离可以很远
- LSTM在长距离依赖上比一般RNN表现好不少，但是没有明确指出长/短上下文表示，而是单独用一个状态进行表示
- LSRC模型核心在于引入Hl和Hg分别抓取局部和全局信息
- 隐变最的时序相关变化性况，可以证明LSTM, RNN, LSRC等模型在对全局和局部信息抓取的性能

### 27. A Study on Neural Network Language Modeling
- 本文对神经网络语言模式(NNLM)进行深入研究，包括重要性采样、分类、缓存、双向循环神经网络，分别从架构和知识表示两方面阐述
- 对于知识表示来说，NNLM一般来说是近似表示词序列在给定词料中的的统计分布，而非语言相关的知识，或者在自然语言中通过这些词序表达的信息
- Basic NNLM: NNLM是一种统计语言模型，也可以称之为神经概率模型或神经统计模型，主要有:FNNLM, RNNLM, LSTM-RNNLM
- Feed-forward Neural Network Language Model(FNNLM): FNNLM没有很好的方法去表示，因此一般都截取了最近的n个词用做输入，使用当前词用做输出
- NNLM一般使用Perplexity衡量性能，其表示用于编码测试数据所要的比特位数，其值越少，代表模型越接最真实的语言情况
- Recurrent Neural Network Language Model(RNNLM): RNNLM不仅关心输入空间，还关心内在状态空间，内在状态空间使表示序列之间的依赖关系
- Long Short Term Memor RNNLM(LSTM-RNNLM): 用于解决长距离依赖和梯度消失等问题
- Importance Sampling: 核心思想是使用采样的方法，近似log-likelihood gradient(可以视为两部分，目标值的正向推动，非目标真的反向推进)；为了防止发散，采样个数需随着训练进行而提升；Importance Sampling可以用于n-gram之类的模型，但不能直接用于RNNLM, LSTM-RNNLM
- Word Class: 每个词被当成一个单独的类，在预测词时，变成了，由历史预测词的分类，再由分类和历史预测词。
- hierarchical neural network language model: 能够加速训练，k/logk, k为所有的词数。
- 此外，分布式词表示可以用于表示词的相似性，但是层级表示会使得这种特征减弱，层数越线，效果越差
- Caching: 缓存语言模型是基于近期历史会有更大可能再次出现这一假设，在这一模型中，条件概率是标准模型和缓存插值；另一种是基于词类的缓存策略；在RNNLM中，通过记录输出和状态，用于未来同样的上下文环境预测
- Bidirectional Recurrent Neural Network: 通过反转输入顺序实现，一个解释是减少了因为跨跃长度而带来的损失，本文的解析是词序列从统计角度看可能更依赖于后续文本，而非之前的内容; 通过实验，可以发现正向和反向几乎可以获取等量的信息; BiRNN可以在翻译和语音识别等问题上实现更好的效果，但其对逐词处理场景并不适用。
- Model Architecture: 许多模型都是逐字进行处理，但是这与人在进行写作和说话的情况并不一样，一般来说，人都提前想好了要说的，或者想好了很大一部分，所以说逐字或者其它的顺序可能是不合理的
- Knowledge Representation: 一个常见的说法是，NNLM从语料中学到了词序列的概率分布，严格来说应该是某一语料中的语序概率分布；总的来说，NNLM的知识表示是词序列在特定训练语料上的分布式表示，无论是语言本身，如语法，还是语言涉及的知识，都不是NNLM能够得到的；NNLM不能动态的从新的数据集中学习知识
- Future: (GRU)RNNLM, dropout strategy, character level neural network language model；自然语言是人创造的，而语言知识是语言出现很久才创造的；语言知识涉及的证确词序，在真实场景中常常被误用；在现实生活中，语言只是声间或符号与抽象或真实物体的连接；因此，由于声音和符号的特片可能比自然语言更容易，可以通过找到声音和符号或物体之间的关系来处理自然语言。

### 28. A Primer on Neural Network Models for Natural Language Processing
- nlp从sparse&linear的模型转换到dese&non-linear模型
- 神经网络的非线性使其容易与word-embedding相结合
- Straight-forward applications of a feed-forward network as a classifier replacement (usually coupled with the use of pre-trained word vectors) provide benefits also for CCG supertagging, dialog state tracking, pre-ordering for statistical machine translation and language modeling.
- Convolutional and pooling architecture show promising results on many tasks, including document classification, short-text categorization, sentiment classification, relation type classification between entities, event detection, paraphrase identification, semantic role labeling, question answering, predicting box-office revenues of movies based on critic reviews, modeling text interestingness, and modeling the relation between character-sequences and part-of-speech tags.
- Recurrent models have been shown to produce very strong results for language modeling; as well as for sequence tagging, machine translation, dependency parsing, sentiment analysis, noisy text normalization, dialog state tracking, response generation, and modeling the relation between character sequences and part-of-speech tags.
- Recursive models were shown to produce state-of-the-art or near state-of-the-art results for constituency and dependency parse re-ranking, discourse parsing, semantic relation classification, political ideology detection based on parse trees, sentiment classification, target-dependent sentiment classification and question answering.
- One Hot VS Dense; CBOW; Feature Combinations; Dimensionality; Vector Sharing; Network's Output
- 非线性激励函数是nn能表示复杂信息的关键
- HINGE: 无论是binary还是multiclass都是线性输出，HINGE LOSS在区分边界很好用，但是对成员概率判断效果有限
- loss function: HINGE(binary, multicalss), log loss, categorical cross-entropy loss, ranking loss
- 非监督的词向量训练方法是：相似的词有相似的向量
- 窗口的大小对词向量的影响：大窗口着重于主题相似；小窗口着重于语法和功能相似
- 词向量与语法信息十分相关
- SGD & Computation graph
- Initialization: xavier initialization, zero-mean Gaussian for ReLU
- Vanishing and Exploding gradients: shallower network, step-wise training, performing batch-normalization, LSTM, GRU, clipping the gradients if their norm exceeds a given threshold
- 如果网络不好使，要关注网络中saturation and dead neurons; saturation: 一般是输入值过大; dead: 梯度过大
- Shuffling & Learning Rate & Minibatches
- Regularization & dropout
- Model Cascading & Multi-task Learning
- NLP的输出一般不是分类标签或者类别分布，而是输出序列，树，或图
- Greedy Structured Prediction & Search Based Structured Prediction & Probabilistic Objective (CRF) & Reranking & MEMM and Hybrid Approaches
- Convolutional Layers: ordered sets of items; CBOW完全取忽略了词序; Dynamic, Hierarchical and k-max Pooling;
- Recurrent Neural Networks – Modeling Sequences and Stacks: backpropagation through time; Acceptor 使用最后的结果作为输出; Encoder最后的信息用作前面所有信息的编码; Transducer使用每一步的输出; Encoder - Decoder; Multi-layer (Stacked) RNNs; Bidirectional RNNs (biRNN); RNNs for Representing Stacks; 
- Concrete RNN Architectures: Simple RNN; LSTM; 
- Modeling Trees – Recursive Neural Networks

### 29. A Convolutional Neural Network for Modelling Sentences
- Dynamic Convolutional Neural Network(DCNN)用于对句子语义建模
- 网络使用动态k-Max Pooling
- 网络可能处理不同的输入长的句子，并产生能够抓住长短距离的关系的特征图
- 算法的核心在于如何从句子的n-gram或词中抽取特征
- a. k-max pooling句子，取得top-k maximun，b. k可以动态选取，可以视为其它特征的函数
- one dimensional filter for each n-gram => k-max pooling and no-linearity => feature map of the sentence
- 高层的small filter可以抓住句子中远距离词之间语法或语义关系
- One-dimensional convolution的基本观点是：使用filter与每个n-gram进行点乘得到另一个序列
- wide convolution 相对于narrow convolution有一定优势
- Max-TDNN pros: 1. 对句子词序敏感，且不依赖外在结构；2. 每个词都有同等重要性；cons: 1. 只能考虑m范围的词语; 2. 当扩大filter或增加层数时，对句子的最小长度有基本要求
- Max-pooling cons: 1. 不能区分相应的feature是否多次出现; 2. 每次pooling过多的删减信息
- DCNN生成的树比语法树更一般化，其不被一些特点的语法约束，而且可以处理长短两种语义关系

### 30. word2vec Parameter Learning Explained
- 对word2vec相关计算原理，过程都有详细的分析介绍

### 31. Hierarchical Probabilistic Neural Network Language Model
- 神经网络在语言模型中有很多成功的应用，其主要的优点在于在连续空间表示词(或者符号)，平滑了语言模型，这样即使在训练数据不足的情况下，也能有很好的泛化效果。
- 相关模型速度相当慢，本文提出基于条件概率的层级分级方法来加速训练，可以使训练和识别加速到200倍
- word classes: 根据词语使用在上下文使用的概率来把词语分成不同的分块.
- 层级分解的基本思想是把线性预测，分解成log(V)步的决策，从而加快了计算; 另一个方法是，使用同一个模型来处理所有的决策
- 基本思想是：不直接计算上下文与某个词的关系时，而是计算上下文预测词所属的类别，再通过类别预测词，从而加速计算
- 每个词可以表示成一个十进制串，大二叉层级树中进行索引
- 采用平衡树，或者huffman tree编码可以进一步加快计算速度

### 32. Bilingual Word Embeddings for Phrase-Based Machine Translation
- 通过学习两个语言对应词的embedding，来优化机器翻译对齐和效果
- 新的词嵌方法，在词语相似方面，显著的提升了效果
- 跨语言的词嵌训练，可以增加跨语言的相似程度
- 通过无监督的方法，来训练双语词嵌，主要是方法是拉开文档中正确的词和随机词的评分间隔

### 33. Training RNNs as Fast as CNNs
- 通过简化状态计算得到更具有并行性的RNN实现

### 34. StarSpace: Embed All The Things!
- 一个通用的神经网络嵌入模型，可以处理: labeling tasks such as text classification, ranking tasks such as in- formation retrieval/web search, collaborative filtering-based or content-based recommendation, embedding of multi- relational graphs, and learning word, sentence or document level embeddings.
- 嵌入由不同离散特征组成的实体，并计算他们之间的相似性。
- StarSpace模型由不同的learning entities组成，每个entity由一组离散的features组成，如词语则可以视为n-gram模型。
- 在StartSpace模型中，任意的实体之间可以比较，即使他们是不同的种类。
- 选对feature进行建模，每个Feature由一个d维向量构成k
- Multiclass Classification: (a, b), (a, b-)，用采样技术生成
- Multilabel Classification: 每个文档有多个标签，从中采成(a, b)对进行训练
- Collaborative Filtering-based Recommendation: 每个用户用其喜欢的物品表示，一个唯一的值表示用用户，用户与其喜欢的物品组成正例，其它的组成负例
- Collaborative Filtering-based Recommendation with out-of-sample user extension: 不用ID表示用户，而是且其喜欢的物品集合(不包括其中一个)，与这一个组成正例
- Content-based Recommendation: 用户由其喜欢的物品描述，物品由其特征描述
- Multi-Relational Knowledge Graphs (e.g. Link Predic- tion)
- Information Retrieval (e.g. Document Search) and Document Embeddings: 监督与非监督两种方式
- Learning Word Embeddings: word2vec
- Learning Sentence Embeddings: 使用来自同一文章的句子为正例

### 35. Think Globally, Embed Locally — Locally Linear Meta-embedding of Words
- 通过结合已有词嵌产生更加精确完善的meta-embedding
- 提出一种无监督算法，使用预训词向量做为输入，生产更加精确的meta-embedding
- 已经提出的词向量拼接，可以看成本文的一种特殊实例
- 面临的问题：训练语料不相同; 输入词向量维度不相同; 不同的词向量中，词的邻近词有很大的不同
- 提出locally-linear meta-embedding学习方法，a) 只需要在词表中的词，b) 可以meta-embed不同长度的源词嵌，c) 对不同词嵌的邻近变化敏感
- 算法主要分为两步：a) recosntuction step: 对每个源词嵌进行细性结合每个词的最邻近词; b) projection step: 计算meta-embedding
- 本方法解决了源词嵌不对齐，以及词语缺失等问题
- 使用BallTree algorithm加速搜索
- 不同的词嵌来源，其优化目标也不一样，这样邻近词表就会进行互补，从而达到更优的效果
- 下一步方向，跨语言meta-embedding实现

### 36. BLEU: a Method for Automatic Evaluation of Machine Translation
- 提出一种自动机器翻译度量方法，具有语言独立、类似人工衡量、并且高效
- 翻译的三个方面，adequacy, fidelity, fluency
- BLUE最基本准则是，对比并统计候选结果与真实结果应用n-gram后匹配的个数
- 1-gram用于保证adequacy, 更长的n-gram增加流畅性
- 在单句上不同的人衡量结果有所不同，BLUE一般用于平均性能
- 对于不存在的词，以及出现过多的词进行惩罚
- 一个词对，应该只依赖一个参考，而非多个
- brevity penality: 关注长度、词语、词序

### 37. Efficient Estimation of Word Representations in Vector Space
- 使用Huffman binary tree做为层级softmax
- CBOW
- Skip-gram
- skip-gram在semantic和syntactic要优于CBOW

### 38. Linguistic Regularities in Continuous Space Word Representations
- 使用输入层权重隐性表示词的向量，可以很好的抓住语法和语义特征，并且每一种特征都具有特征空间偏移特性
- 训练一个网络，一方面可以得到模型本身，另一方面，可以得到词的表示
- 使用网络训练，由上文预测下文
- 提供一个offset衡量分布式表示的方法
- 所有结果，以及附加品，都是无监督学习取得
- 基于cosine distance的偏移测量
- 语义和语法测试集

### 39. WORD TRANSLATION WITHOUT PARALLEL DATA
- 不需要平行语料，只需要单一语言语料
- 训练分为两步，a. 对抗训练，把两种语言映射到目标空间，b. 抽取合成词典，转化成the closed-form Procrustes solution
- 提出一种与映射质量高相关的非监督筛选标准
- 给出12种语言的词典，并给出相应的监督与非监督训练对
- 使用CSLS生成词典
- 由于低频词效果有限，只训练top 50000词
- 保证W矩阵接近正交，可以增加效果
- 计算翻译的平均cosin距离，用作验证标准

### 40. Hierarchical Probabilistic Neural Network Language Model
- 层级结构

### 41. DisSent: Sentence Representation Learning from Explicit Discourse Relations
- 使用'because but, although等词来标识句子之间的关系。
- 使用双向lstm encode句子，然后进行时域maxpooling，然后向量相加
- 使用依存分析去掉一些discourse使用

### 42. Learning Distributed Representations of Sentences from Unlabelled Data
- 提出一种新的句子或短语训练目标- Sequential Denoising Autoencoders(SDAEs)和FastSent
- 不使用botton up策略，pharse或sentence表示是在词向量的数学运算的基础上
- k

### 43. Advances in Pre-Training Distributed Word Representations
- 提出一种新的词向量预训方法
- 几种很少一起使用的策略：the position dependent features，the phrase rep- resentations，the use of subword information
- 原始word2vec只是简单的平均窗口中的词，并没有考虑位置信息
- 原始cbow只是unigrams，没有考虑到词序
- 标准的词向量忽略了词的内部结构

### 44. Zero-Shot Relation Extraction via Reading Comprehension
- 一般的方法不能抽取未提前说明或没在训练中见到过的实体
- 在数据和模型都提出了新方法，1) 使用距离监督来处理巨量的关系，2) 使用众包的方法来收集和验证每一个关系的问题
- 在解决阅读理角问题上，与解决关系抽取填空问题存在一定的相似性
- 问题的难点在于把关系转化成问题，而非处理每个实体的答案，这样在标时也从实体标转化成为关系标注，提升了效率
- 第一个schema level基于众包的qa数据集

### 45. Neural Relation Extraction with Selective Attention over Instances
- 远程监督关系不可避免的伴随标注错误问题
- 提出句子级基于注意力机制的关系抽取方法
- 1) 相比于现有的神经网络关系抽取，本模型可以充分利用实体间的信息，2) 使用选择性注意力来降低noise的影响，3) 实验结果显示有效
- 与别的模型不同的是，提出在多个实体上的句子级注意力机制，可以利用所有的信息
- 模型分为两个阶段：1) 学习句子的分布式表示，2) 使用句子级注意力机制选取表达关系的句子

### 46. DIALOGUE LEARNING WITH HUMAN-IN-THE-LOOP
- 一般的对话机器人训练着重于在固定数据集上训练，本文使用强化学习基于在线反馈训练在线模型。
- 本文考虑两种反馈：明确的奖励，文本回应
- 本文的模型policy可以视为在数据集上的多次迭代
- 数据测试分为两步，1) 模拟环境，2) Mechanical Turk平台
- 本文只考虑，Task 6，当BOT回答正确时，Teacher给出正面回答，当机器人回答错误时，Teacher给出文本答案。
- 底层网络使用MemN2N
- 1) 转化文本为向量表示，2) 转化记忆为向量表示，3) 选出与文本相关的记忆单元，4) 记忆可以进行多次查询，5) 使用最后查询向量表示与候选者进行softmax，得到相应概率
- policy: MemN2N, state: 聊天记录，action: MemN2N结果，reword: 对(1),错(0)，
- 使用batch size来确定模型更新时间

### 47. 360° Stance Detection
- 一个使用自然语言检测言论风向的应用

### 48. Investigating Capsule Networks with Dynamic Routing for Text Classification
- 使用了三个方法来减轻capsules的扰动，并在6个任务中4个达到了较好的结果
- 使用capsules的向量输出，来代替CNN的标量输出，用于保证局部的词序信息。

### 49. THE UNREASONABLE EFFECTIVENESS OF THE FORGET GATE
- provided another network just use the forget gate
- half parameter and two-third of the element-wise multiplications.
- the key is the parameter initialization

### 50. Scalable attribute-aware network embedding with localily
- 网络词向量可以方便联合学习拓扑和属性
- linearity and scalable

### 51. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
- 本文发现，每简单的卷积网络都可以超过权威的循环神经网络。
- 有许多论文对各种循环神经网络进行过验证。
- 本文结合Causal Convolutions、Dilated Convolutions、Residual Connections得到TCN
- TCN的主要优点有：并行、灵活的接收域、稳定的梯度、训练占用内存少、可变长输入。
- 本文在The adding problem、Sequential MNIST and P-MNIST、Copy memory、JSB Chorales and Nottingham、PennTreebank、Wikitext-103、LAMBADA、text8等多个数据集上进行试验，与常见的LSTM、GRU等网络进行对比，在结果和性能上TCN都取得了相当不错的优势。

### 52. Attention is all you need
- Scaled Dot-Product Attention
- Multi-Head Attention
- Using dk to pushing the softmax function into regions where is has extremely small gradients.
- Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
- Positional Encoding: position embeding and sinusoidal

### 53. OpenSeq2Seq: extensible toolkit for distributed and mixed precision training of sequence-to-sequence models
- seq2seq, and fully supporting ditributed and mixed-precision training
- any encoder can combined with any decoder

### 54. Deep contextualized word representations
- ELMo representations are deep, in the sense that they are a function of all of the internal layers of the biLM
- we show that the higher-level LSTM states capture context-dependent aspects of word meaning,  while lower level states model aspects of syntax
- Our approach also benefits from subword units through the use of character convolutions, and we seamlessly incorporate multi-sense information into downstream tasks without explicitly training to predict predefined sense classes.
- syntactic information is better represented at lower layers while semantic information is captured a higher layers

### 55. Sliced Recurrent Neural Networks
- 在不改变RNN单元的情况下，加速RNN的运行速度
- 只针对RNN的速体结构进行改变，并不改变局部运行情况，这对单元没有特别的要求
- 在6个大数据集上进行训练，取得很不错的结果
- 把每层分成n份，在有需要时进行padding
- 在某些参数设置的情况下，SRNN和RNN是等价的

### 56. Recent Trends in Deep Learning Based Natural Language Processing
- The main advantage of distributional vectors is that they capture similarity between words
- Another limitation comes from learning embeddings based only on a small window of surrounding words, sometimes words such as good and bad share almost the same em- bedding
- the performance of each network depends on the global semantics required by the task itself.

### 57. FASTER: An Embedded Concurrent Key-Value Store for State Management
- 证明了访问的时间关联
- 无锁并发哈希索引：一种支持内存和主存的并发日志结构，同时支持内存就地更新
- 专注于：云应用和状态存储可以在低开销下将状态管理深度集成到高级语言逻辑中，

### 58. Universal Transformers
- 基于Transformer，但是不是一个位置一个位置的迭代，而是整体的对序列进行修改
- 模型的计算瓶颈不在于序列的长度，而在于改写的版本数
- 解码的时间，只能是其左边的位置

### 59. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- BERT使用双向表征训练网络，同时在每一层考虑左右两边的context
- 使用预训表征主要有两种方法，即：feature-based, fine tuning
- 使用masked language model
- 使用bidirectional transformer做为encorder，left-context-only transformer做为decoder
- 每个token的输入是token，segment，position Embedding
- 使用msked words，使用策略：80%替换为[MASK]，10%替换成随机词，10%保持不变
- 由于transformer不知道哪个词会被替换，因为他被强制保持了每个词的表征

### 60. Improving language understanding with unsupervised learning
- 使用transformer
- 使用无监督语料进行预训
- 使用辅助loss
- 使用简单易用的task transfer框架

### 61. XNLI: Evaluating Cross-lingual Sentence Representations
- 生成多种语言的翻译，从而控制了主题
- 通过多语言表示的词向量的平均进行比较，固定英语表示，训练其它的表示

### 62. pair2vec: Compositional Word-Pair Embeddings for Cross-Sentence Inference
- 训练word embedding并得到pair word embedding
- 训练的基本思路是提升互信息
- 从top词典中进行sample，用于打压普遍信息

### 63. Stochastic Adaptive Neural Architecture Search for Keyword Spotting
- 提出自动适应算法，提升了识别速度，而且保持精度
- 在架构中加入对性能损耗的预估

### 64. TRANSFORMER-XL: ATTENTIVE LANGUAGE MODELS BEYOND A FIXED-LENGTH CONTEXT
- LSTM存在信息丢失情况
- Transformer-XL每个新的段落不从初始进行训练，而是从上一个段落的基础上进行训练
- 使用相对位置编码，而非绝对编码，可以防止时间混淆
- 暴力解法是把文本分成一段段

### 65. What do you learn from context? Probing for sentence structure in contextualized word representations
- 验证了预训模型相对于word2vec等模型，在语法、语义、局部、全局的表现

### 66. Language Models are Unsupervised Multitask Learners
- GPT-2

### 67. R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS
-  问题和上下文首先分别输入，然后进行MATCH

### 68. Multilingual Constituency Parsing with Self-Attention and Pre-Training
- 引入pre train，并对比了ELMo和BERT

### 69. Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation
- 为了加速，encoder的底层接入decoder的顶层
- 为了处理OOV，使用subwords
- 使用了length normal的beam search
- 一般来说NMT在三个方面弱于Pharsed based: a. 速度，b. OOV效率, c. 失败率
- 如果不使用length normalization，解码倾向于短的句子

### 70. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
- trivia qa dataset

### 71. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset
- 阅读理解数据集

### 72. To Tune or Not to Tune?  Adapting Pretrained Representations to Diverse Tasks
- 通过对bert和elom的利用和细调对比，发现在与预训任务相差大和不大的情况下，效果差不多。
- ELMo在处理序列之间的关系存在问题
- 额外参数对frozen有较大影响，对fintune没有很大影响
- fintune对参数有一定的要求
- 域之间没有多大的影响
- 随着层的提升，表征的质量也在不断的提升

### 73. Text Classification Algorithms: A Survey
- 降维比开发新算法，更有效

### 74. Generating Long Sequences with Sparse Transformers
- 使用不同的PATTERN来加速全连接矩阵计算
- 更多的head有助于提升准确度

### 75. The Curious Case of Neural Text Degeneration
- 使用beam search会极大的使生成模型退化
- Nucleus采样，有效的降低了从不可靠的长尾分布采样的风险
- transformer可能有导至loop的倾向

### 76. VideoBERT: A Joint Model for Video and Language Representation Learning
- combined bert and video task

### 77. Analogies Explained: Towards Understanding Word Embeddings
- explain why if “wa is to wa∗ as wb is to wb∗” then a linear relationship manifests between correponding word embeddings.
- 从PMI的角度来解释W和C的关系，如果W和C等价，也代表PMI为半正定矩阵，这对很多语料是不满足的

### 78. XLNet: Generalized Autoregressive Pretraining for Language Understanding
- big transformer network

### 79. Language Modeling with Gated Convolutional Networks
- 一种高速且更有效的语言模型结构

### 80. Chinese NER Using Lattice LSTM
- sota

### 81. Unsupervised Question Answering by Cloze Translation
- 结合使用完形填空和翻译的方法完成QA生成
- 结合使用无监督的端到端翻译方法

### 82. MULTIPLE-ATTRIBUTE TEXT STYLE TRANSFER
- 分离不是控制变量的因素，即使对搞LOSS也不是
- 使用池化来衡量风格转换和内容保持
- 支持多特点控制
- 使用bias来控制生成的风格

### 83. Recursive Regularization for Large-scale Classification with Hierarchical and Graphical Dependencies∗
- 一般的条件方法，在小的数据集已经证明有效，但是存在大规模扩展的局限性
- 方法能应用到不同的分类方法，同时也能充分利用层级或者图关系
- 每个不同的区域可能并行的进行

### 84. Large-Scale Hierarchical Text Classification with Recursively Regularized Deep Graph-CNN
- 在主题分类中，可能更需要考虑的关键词、句等因素
- 基于递归正则的方法

### 85. Adaptive Attention Span in Transformers
- 在底层只需要局部信息就足够了
- 自动适应Trasnformer的长度

### 86. A NEW METHOD OF REGION EMBEDDING FOR TEXT CLASSIFICATION
- 提出一种新的n-gram表征方法
- 每个词向量表生分为embeding和一个局部基于权重context表征
- 基于region得到word-context和context-word两种向量表示

### 87. Hierarchical Attention Networks for Document Classification
- 使用层级attention进行文本分类
- 使用两个级别的attention，分别是词级别和句级别

### 88. Large Memory Layers with Product Keys
- 定义一个key set，每个key有相应的内存插条
- 稀疏参数选取和更新，加速计算
- 基于knn搜索
- 12层的bert超过了24层的bert
- 使用小的vector表征大的vector
- 所有操作都是可微分的，保证了在网络是的可插拔
- 为了保正KNN计算的快速性，使用了product key
- 对于稀疏矩阵的update，保持一下较高的learning rate是个不错的选择

### 89. Attentive Convolution: Equipping CNNs with RNN-style Attention Mechanisms
- 使用attentions让CNN不仅得到局部的信息，还有非局部信息
- 非局部信息来源：1)相对远处的，2)额外信息
- 在多层CNN中，直接feed tx or ty效果不好

### 90. Deep Pyramid Convolutional Neural Networks for Text Categorization
- CNN中词向量优于字向量
- 增加网络的深度，但是不增加网络的计算复杂度
- 深度网络可以发现更多长距依赖
- down sampling, shortcut and text region embedding
- 增加feature map并没有增加精度
- 在shortcut中，不进行维度改变

### 91. Effective Use of Word Order for Text Categorization with Convolutional Neural Networks
- 实验证明seq-CNN在情感分类任务上，优于bow-CNN
- 使用多种pooling

### 92. Convolutional Neural Networks for Text Categorization: Shallow Word-level vs. Deep Character-level
- 从已有的结果上可以行到，word-CNN在小数据集上要优于char-CNN

### 93. Efficient softmax approximation for GPUs
- 使用adaptive的方法，来处理巨量softmax矩阵
- 自然语言数据分布为Zipf定律

### 94. A Simple Theoretical Model of Importance for Summarization
- summary不应该包括太多的信息
- summary是最小化KL，同时最大化Relevance，最小化Redundancy
- summary是有效信息量

### 95. PREDICTING THE GENERALIZATION GAP IN DEEP NETWORKS WITH MARGIN DISTRIBUTIONS
- 深度神经网络可以很好的拟合随机数据，但是泛化有限，这证明一般的loss function不足发很好的代表泛化性
- 跟一般的方法的不同之处
  - 跟weight normal相关方法不同之处在于，对网络结构变化感知，而且不局限于网络结构
  - 在一两层设置gap是不够的
  - 一般的间距设置是指训练集和决策面之间的距离
  - 仅仅是间距的设置，并不一定能达到提升泛化的效果
- 如果一些层次跟输入层相差很大，也证明他们在泛化上的贡献很大

### 96. Distilling the Knowledge in a Neural Network
- 在线上服务时，可以用蒸馏的方法来提取大模型中的知识，以达到模型服务加速的效果
- 使用soft target相比于hard target可以提供更多的信息，以及在gredient中更少的variance
- distill: 提升softmax的温度，直到模型产生合适的soft target，再用这些soft target去训练简单模型
- 在训练小模型时，可以使用一样的数据或者完全没有标签的数据
- 训练小模型同时使用hard target和soft target可以有更好的结果
- soft obj仍然使用原模型的high temp loss，hard obj使用正常的，并对soft obj加权T^2

### 97. Directional Skip-Gram-Explicitly Distinguishing Left and Right Context for Word Embeddings
- 在词向量训练中引入方向向量

### 98. On Extractive and Abstractive Neural Document Summarization with Transformer Language Models
- 把摘要分为抽限和总结两部分
- 抽取摘要获取的方法：选择文本中与摘要ROUGE score最高的几个句子
- 使用ground truth extract在训练时期，在inferenece时使用抽取的，要比都使用抽取的好
- 使用topk sampling

### 99. ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS
- 更大的模型，有更低的mask acc，但是并没有过拟合
- 减少参数的方法：a. 对embedding table进行分解，b. 多层参数共享
- 减少参数的情况下，参数量减少了18倍，训练速度提升了1.7倍
- 引入sop，进行句子级别控制
- NSP问题，混淆了主题预测和一致性预测
- 引入SOP能极大的提高
- 添加额外数据，能提高精度
- 无dropout也能提高精度
- dropout可能对transformer-based模型，有负面影响 

### 100. Deep Equilibrium Models
- 匹敌的效果
- 算力不变
- 内存压缩
- 在很多的应用中，序列模型常常证明是大于深度模型的
- 可以压缩多层，达以一致效果

### 101. SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Cross-lingual Focused Evaluation
- 检测语义相似度相关任务

### 102. Training Deep Nets with Sublinear Memory Cost
- 使用一些额外的计算开销，来达到减少模型内存的目的
- 内存减少到O(logn), 计算至O(nlogn)
- 主要集中在减少存储中间变量的开销
- 优化计算机程序和优化深度网络十分类似
- 只有在生命周期不重合的权重才能共享内存
- 对部分变量在使用时进行再计算
- 把模型分成不同的段，只记住每段的输入和输出，对每段进行重计算

### 103. ERNIE 2.0: A CONTINUAL PRE-TRAINING FRAMEWORK FOR LANGUAGE UNDERSTANDING
- 引入多种任务学习，包括：字、结构、语义等级别
- 词级别任务：a) 字、实体mask、实体识别，b) 大写词识别，c) 预测词是否在文本中其它的部分出现
- 结构级别任务：a) 把句子随机分成n个部分，预测order，b) 句子距离预测，0相邻，1同文档，2不同文档
- 语义级别任务：a) 语义关系，b) 使用搜索引擎数据

### 104. RoBERTa: A Robustly Optimized BERT Pretraining Approach
- 发现bert没有训练到位，还有很大的提升空间
- 1) 更长的序列，更大的bz，和更多的数据，2) 去掉下一句预测，3) 在更长的序列上训练，4) 动态改变mask
- 对于adam，使用beta2 = 0.98在大batch size下更稳定

### 105. SpanBERT: Improving Pre-training by Representing and Predicting Spans
- 进行连续范围的mask，而非进行个别mask
- 训练区别边界去预测整个区间，而非个别token
- 不使用NSP，可取得更好的结果
- 同时使用边界loss和mlm loss

### 106. LARGE BATCH OPTIMIZATION FOR DEEP LEARNING: TRAINING BERT IN 76 MINUTES
- 在covex model中，同步训练大batch size可以因为方差的减少而受益，一般而言在大的batch size时，可以相应的方大lr到`sqrt(news_batch_size/old_batch_size)`
- 在noncovex model中，线性增长lr，一般需要使用warm up，线性增涨lr，在超过一定的batch size时，会变成负收益
- 本文主要集中在多层网络
- 基于每一层l2 norm；每一层的lr被放大至f(||xt||)
- 在bert的第二阶段进行re warm up

### 107. MEGATRON-LM: TRAINING MULTI-BILLION PARAMETER LANGUAGE MODELS USING MODEL PARALLELISM
- 更大的模型
- 在512个GPU上，达到76%的扩展系数
- SOTA结果
- 基于列分解计算

### 108. LARGE-SCALE PRETRAINING FOR NEURAL MACHINE TRANSLATION WITH TENS OF BILLIONS OF SENTENCE PAIRS
- 增加数据，不一定GPU能处理
- 增加数据也意味着更多的噪声
- 数据来源很杂乱，使用预训，调整方法
- 使用dynamic data split

### 109. Knowledge Enhanced Contextual Word Representations
- 提出一种把知识库嵌入到大规模模型中的方法
- 每个entity span召回一系列的候选

### 110. Extreme Language Model Compression with Optimal Subwords and Shared Projections
- 压缩student模型的词表大小
- 在base model达到了60x的压缩
- 使用更小的词表
- nlp中的模型压缩通常包括：矩阵近似，参数裁减/共享，权重量化、知识蒸馏
- 随机选出一些词，用teacher分割，另一些词用student分割
- 使用映射矩阵进行对齐

### 111. TinyBERT: Distilling BERT for Natural Language Understanding
- 设计一种适合多stage的蒸馏方法
- 取得了相当的结果

### 112. Get To The Point: Summarization with Pointer-Generator Networks
- 抽象摘要可能会导致两个问题，a. 不准备的表述；b. 重复
- 1. 使用pointing来拿取原文中的信息，2. 使用coverage来跟踪摘要，保证不重复
- 使用coverage来降低重复

### 113. Pointer networks
- 可以提供一个可变的词典
- 主要贡献
  - 提出一种基于"pointer"的可变词典的生成架构
  - 应用到各种问题
  - 完全基于数据驱动

### Incorporating copying mechanism in sequence-to-sequence learning
### Language as a latent variable: Discrete generative models for sentence compression

### UNITER: Learning UNiversal Image-TExt Representations
### K-BERT: Enabling Language Representation with Knowledge Graph
### Language Models as Knowledge Bases?
### Investigating BERT's Knowledge of Language: Five Analysis Methods with NPIs
### EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks

### Word Mover’s Embedding: From Word2Vec to Document Embedding
- 

### Scaling memory-augmented neural networks with sparse reads and writes



### Sentence Mover’s Similarity: Automatic Evaluation for Multi-Sentence Texts
- 

### Language Models are Unsupervised Multitask Learners
- 

### Learned in translation: Contextualized word vectors

### Character-Level Language Modeling with Deeper Self-Attention


### Adversarial training methods for semi-supervised text classification

### 57. Language Modeling with Gated Convolutional Networks

### 57. Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification∗
1. 情感分析专用词向量

### 58. A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning

### 58. “Beyond bilingual: Multi-sense word em- beddings using multilingual context,”

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

### Deep contextualized word rep- resentations
### Neural machine translation in linear time
### Language Modeling with Gated Convolutional Networks
