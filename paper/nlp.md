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
