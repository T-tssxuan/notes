## NLP related paper

#### [1. GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
1. 使用全局非0 word-word cooccurrence matrix进行计算向量信息
2. LSA 丢失了文档词序
3. 滑动窗口等方法没有利用全局信息
4. 最大拟合词之间向量积和统计的共同出现信息
<img src="http://chart.googleapis.com/chart?cht=tx&chl= ">\widehat{J}= \sum_{}^{i,j}f(X{i,j})(w{T}^{i}\widetilde{j}{j}-\log X{ij})^{2}" style="border:none;">
5. 相对于CBOW, n skip-gram交果要提升很多
