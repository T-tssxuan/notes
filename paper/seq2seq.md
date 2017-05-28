## seq2seq related paper

#### 1. Self-critical Sequence Training for Image Captioning
1. 在image caption上使用强化学习
2. SCST不使用常规的基准激励函数，而是自己输出来进行推导激励
3. non-differentiable and exposure bias这两个问题使用reinforcement learning都能得到很好的解决
4. reinforcement learning的variance十分高，如果没有适当的context-dependent normalization，非常不稳定
5. SCST不是去衡量reward或者normalize reward signal，其使用其测试时使用的算法来回馈其行为
6. 使用attention model去处理输入图像的不同位置

#### 2. Convolutional Sequence to Sequence Learning
1. 介绍一种基于cnn的神经网络处理seq2seq

#### 3. Massive Exploration of Neural Machine Translation Architectures
1. NMT是一种end-to-end自动翻译方法
2. 很多流行的NMT方法是基于ENCODER-DEODER机制的
3. 一些建议：deep encoder比decoder难以训练；密集残差网络比常规的残差网络更好；LSTM比GRU好；精细的调节好的beam search对获取好的结果很重要
4. 生成时，每个符号的值取决到其前面的值和context vector(attention vector)
5. 我们以为更大的embedding可以得到更好的结果，但是并不是这样的

#### 4. pix2code: Generating Code from a Graphical User Interface Screenshot
1. 针对三种设备：ios, android, web
2. 使用RNN和CNN
3. 主题可以分成三部分：1. 使用CV理解给出的场景，推断物体特点、位置等；2. 对需要理解的文本进行语言建模，包括语法上和语议上的正确解析；3. 结合1和2的结果通过抽取场景的隐含信息来理解并给出文本描述
4. 样本在传入CNN前都被转化成256 X 256的图像
5. 使用定量的token，使得可以使用softmax进行选择
