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

### 8. TextRank: Bringing Order into Texts
1. Graph-based 排名算法，是一种基于全局信息对结点进行递归的排序算法。
2. 正如Graph-based 排名算法，投票越高的词其重要程度也越高
3. 使用无向带权图，定义不同的结点之前的关系
4. keyword extraction抽取文中最重要的词
5. 使用co-occurrence定义词之间的关系
6. sentence extraction抽取文中最重要的句子，使用similary来定义句子之间的相似度
7. 进行摘要

### 9. Component-Enhanced Chinese Character Embeddings
1. 基于汉字的边旁部首进行词向量提取

### 10. Character-level Convolutional Networks for Text Classification∗
1. 文本分类主要包括文本feature抽取，和分类器设计
2. 学习时，不需要语义和语法上的先验知识
3. 使用pooling可以最高计算的尝试为6 conv layer and 3 fully-connected layer
4. 使用定长的数据
5. 使用data augmentation 可以提高深度模型的范化结果，但是在文本处理中，需要考虑到其顺序，因此，采用近义词典，对词进行替换，可能得到比较好的结果。
6. Bag-of-means is misuse of word2vec

### 11. FASTTEXT.ZIP: COMPRESSING TEXT CLASSIFICATION MODELS
1. 丢失可接受的精度来获取更快的速度
2. 使用压缩，去除不重要的词
3. LSH，使用binarization strategy 进行cosin相似度搜索
4. proposed approach 提到对于不对称信息分类的一个好的方法
5. 使用PQ进行搜索，k和b的选取，使用搜索可以非常快
6. 选取一些重要的feature，并保证每个文档都被覆盖

### 12. Distributed Representations of Sentences and Documents
1. 提供一种定长向量表示句子的方法
2. 连接多个向量，求得句子向量，并由此求句子之后的单词
3. inspired使用每个词周围的词向的拼接或者均值做为当前向量的值
4. 使用huffman tree进行hierarchical softmax
5. 把句子的向量也当做词语分类的一部分
6. 保持了句子的顺序信息

### 13. A Deep Reinforced Model for Abstractive Summarization
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

### 14. Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts 2014
1. Character to Sentence Convolutional Neural Network(CharSCNN) 使用两个CNN去抽取任何大小的字符和句子的特征
2. 从character-level到sentence-level提取特征
3. Word-Level Embedding: extract syntactic and semantic information
4. Character-Level Embedding: morphological and shape information
5. 句子级别的有两个问题：句子存在不同的长度；句子的重点会出现在不同的位置

### 15. Incremental Skip-gram Model with Negative Sampling
1. 现在的词嵌算法，包括SGNG是multi-pass algorithms，因些不能进行增量模型更新
2. 分析证明，在无限数据集的情况下，increament SGNG算法效果与origin SGNG算法性能接近
3. 自适应算法更适用于增量试训练，因为数据量之前是不知道的，而且可能一直不断的增加
4. 记录TOP K个经常出现的词，这些词是一个在这K个数据中是动态出现的

### 16. Learning to Identify Ambiguous and Misleading News Headlines
1. 准确的定义问题，以及对模糊和误导进行分别对待，使用class sequential rules提取信息来处理模糊定义，使用标题和文本结合来处理误导倾向的标题
2. 使用co-training, semi-supervised method等方法来训练模型
3. 新闻标题可以分为三类：全名合理的、模糊不清的、以及误导的
4. 对于误导性标题，提取独立于正文的特征、依赖于正文的特征，并使用co-training进行训练

### 17. A Deep Network with Visual Text Composition Behavior
1. 一种新型网络，可以层级表句子的组成，还能显示低层如何关注于单个单词的
2.  Attention Gated Transformation (AGT)，每一层通过获取更多的词和句子，新的信息和之前的信息结合，产生新的表示
3. Gate由Attention控制，但是每层的结果会被上层影响
4. 分析attention和gate的分布，可以发现网络在合成词和句子的过程

### 18. Efficient Vector Representation for Documents through Corruption
1. Doc2VecC使用简单的平均词向量，并加上正则项来提取文档语义。
2. 正则项主要作用为：提升稀有或信息量大的词，并使没有区分意义的词接近0
3. Doc2VecC产生比Word2Vec更有意义的词向量
4. Doc2VecC在训练文档表示过程中同时训练词向量表示
5. Doc2VecC在训练过程中随机移除文档中的词，从而加快训练速度
6. 训练复杂度只随词的增长而增长，与文档多少无关
7. 使用Taylor Expansion对损失函数进行逼近
8. 可能在长文本处理上，效果更好

### 19. A Semantics-Based Measure of Emoji Similarity
1. 通过词嵌模型分析emoji的语相似度
2. EmoSim508数据库
3. 主要的用处：1. 检索、2. 情感分析、3. 手机键盘分析
4. deprecated

### 20. High-risk learning: acquiring new word vectors from tiny data
1. 从已有表示上学习新的词嵌表示
2. 需要从小数据中获取词向量的主要原因是：a. 一些词语非常不常见，b. 快速取得新的词的词意
3. 模拟人遇到生词的过程，从之前的词语中能个推断出当前词语的意义
4. 通过从词典中取出新词的最近K个词，来测试训练效果
5. 初始为context的和、高Learning rate、large window size

### 21. Challenges in Data-to-Document Generation
1. 随着输入的增长，输出结果的误差情况越来越严重
2. 从数据表中生成句子的主要特性有：a. 很容易生成与数据集对应的摘要，b. 摘要主要专注于覆盖数据库中的信息
3. 神经网络系统能够产生较流利的输出，而且在词级的匹配上效果也很不错，但是其内容选择和长文本处理上有所欠缺
4. 数据集提供record和描述
5. 三个标准: Content Selection(CS), Relation Generation(RG), Content Ordering(CO)

### 22. On the State of the Art of Evaluation in Neural Language Models
1. 本文揭示了一些常见的
2. Capacity and Trainability in Recurrent Neural Networks提供了一些关于rnn网络能力跟参数个数之间的关系

### 23. Learning to Ask: Neural Question Generation for Reading Comprehension
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

### 24. Effective Dimensionality Reduction for Word Embeddings
1. 对训练好的词向量，进行再进一步的处理，如降维，以适应低配设备
2. 首先对词向量进行中心化，然后使用pca比较
3. 由于向量中，信息可能被一些主导，因此除去主导信息，可以提升词向量质量
4. 发现在ppac算法之后，主导域重现，再次使用ppac有很大的必要
5. 在PPA一文中发现，词向量(包括Glove, word2vec)均值向量偏大，而且大部分能量都集中在非常小的子空间中，如果移除高能子空间的影响，可以提升词向量质量。

### 25. Long-Short Range Context Neural Networks for Language Modeling
1. 语言模型主要是为了抓取语料的统计和结构特征，短范围依赖一般用于提取语言语法特征，长距离依赖一般用于语言语义提取
2. 本文提出一种多尺度架构，本方法分别处理长和短上下文信息，并且动态的结合完成相应的任务
3. 使用Long-Short Range Context(LSRC)网络实现，其同时使用局部(short)和全局(long)上下文，并使用两个隐层表示
4. 一般来说n-gram模型不用于抓取长距离依赖，本文定义基于距离d的关联分函数，Pd(w1, w2)/(P(w1)xP(w2))，如果大于1，证明w1, w2在距离为d情况下，存在关联，通过实验，发现关联是常见的，而且跨跃的距离可以很远
5. LSTM在长距离依赖上比一般RNN表现好不少，但是没有明确指出长/短上下文表示，而是单独用一个状态进行表示
6. LSRC模型核心在于引入Hl和Hg分别抓取局部和全局信息
7. 隐变最的时序相关变化性况，可以证明LSTM, RNN, LSRC等模型在对全局和局部信息抓取的性能

### 26. A Study on Neural Network Language Modeling
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

### 27. Hierarchical Probabilistic Neural Network Language Model
1. 

### 28. A Primer on Neural Network Models for Natural Language Processing
1. 

### 22. Recurrent neural network based language model
