## Neural Machine Translation Related Paper

### 1. Neural Machine Translation and Sequence-to-sequence Models: A Tutorial
1. Word-by-word Computation of Probabilities
2. MLE方法保证了对训练数据的忠诚度
3. 使用插值(interpolation)结合不同统计模型
4. log likelihood: 1. 因为模型数字可能会很少，这样会导致精度缺失; 2. 使用log likelihood可以使得计算更加容易
5. Perplexity: 我们依据语言模型随机概率分布随机选择词，平均有多少词选择是正确的。
6. Unknown Words: Assume closed vocabulary; Iterpolate with an unknown words distribution; Add an <unk> word;
7. Calculating features: 上下文中的信息，可能对预测有用。
8. Selection preferences: what will do what to what.
9. vanishing gradient & exploding gradient
10.  Greedy 1-best Search, Beam Search,

### 2. OpenNMT: Open-source Toolkit for Neural Machine Translation
1. 完全基于sequence-to-sequence实现
2. 包括诸如：multi-layer RNN, attention, bidirec- tional encoder, word features, input feeding, resid- ual connections, beam search, and several others.

### 3. Neural Machine Translation by Jointly Learning to Align and Translate
1. 常规的encoder-decoder模型，都使用固定长度向量表示输入句子，本文推测这可能是encoder-decoder模型的瓶颈之一，本文提出来扩展方法，使得模型能够自动从源句子中搜索与目标词相关的部分，而非进行硬分割。
2. 如何把不同长短句子的信息压缩到一个固定长度的向量中去，是常规encoder-decoder模型需要考虑的一个问题
3. 本文提出的扩展模型，不需要网络把整个句子的信息嵌入信息，而是在解码时自动搜索源句子
