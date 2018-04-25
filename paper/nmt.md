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

### 4. Sequence to Sequence Learning with Neural Networks
1. 使用LSTM构建一个sequence-to-sequence网络
2. 把源句子倒置，可以得到更好的结果，因为这样拉近了很大一部分词语之间的距离
3. DNN适用于输入输出可以被编码成固定长度的向量，但而对不定长的输入输出，这是一个很大的局限性
4. 架构的三个特点：a. 输入和输出使用两个不同的网络; b. 使用深度LSTM网络; c. 倒置输入句子

### 5. Unsupervised Machine Translation Using Monolingual Corpora Only
1. 调研不使用平行语料进行翻译训练
2. 我们假设每一种语言都存在一个单语言的语料库，两个原因：a. 对于未标记语言能进行标注，b. 提供了一个强的性能下界
3. 主要方法是构建两种语言间共用隐空间和基于两个原则的重构进行翻译，两个原则是: a. 从句子的失真版进行重构，b. 模型能够从句子的目标空间的带噪声翻译重构原始句子
4. 除了重构目标约束，使用对抗正则项约束源和目标句子的隐空间表示分布一致
5. 为了使其完全无监督，模型初始化为逐字翻译，语汇表也是从模型语言中学到
6. 提供一个encoder和一个decoder，分别负责编码到潜在空间和解码到源语言或目标语言，但是不同的语言使用不同的lookup table
7. y0用于标志是哪一种语言的开始
8. 通个迭代，直到使模型收敛
9. 使用auto-encoder会使模型很容易收敛，但是只是identity copy，模型并没有学到什么
10. DAE两种加噪方式：a. 丢弃句子中的部分词, best p=0.1，b. 调换名子中词语的顺序, best k=3
11. 对各种语言encoder都能编码到相同的空间，这样方便decoder进行解码
12. total loss: auto_src + auto_tgt + cd_src_tgt + cd_tgt_src + adv
13. 在100000个平行对时，训练的效果跟supervised方法近似

### 6. Phrase-Based & Neural Unsupervised Machine Translation
1. back-translation key idea is to maintain two medels, one for translating the source into the target and the other to translate the target to the source.
2. pharse based translation system is better than neural system when labeled data is scare.
3. We claim that unsupervised MT can be accomplished by leveraging three components : suitable initialization, language modeling and iterative back-translation
4.Initialization with BPE code; Language Modeling is a denoising autoencoding; Back-translation; Sharing Latent Representations is used to constraint the result(share the encoder parameters and decode parameters); compare with others we removed the adversarial term, and simplify the model.
5. NMT for the language share many things language, the PBSMT used to train the languages shares rarely information
