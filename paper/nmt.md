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

### 2. Neural Machine Translation by Jointly Learning to Align and Translate
