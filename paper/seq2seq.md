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
2. 
