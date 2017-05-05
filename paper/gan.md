## GAN related paper

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

