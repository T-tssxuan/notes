## CNN related paper

#### 1. Deep Residual Learning for Image Recognition(Residual Net)
1. 原始问题变成F(x) = H(x) - x
2. 网络复杂度低于VGG net
3. 有很高的泛化，可以用于视觉和非视觉
4. 使用y = F(x, {Wi}) + x没有添加额外的复杂度，也没有增加过多的计算量
5. F函数可以很灵活，两到三层都可以，但如果只有一层，可能不怎么适合


#### 2. ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
1. top-1 and top-5 error rates of 37.5% and 17.j0%
2. 60 million parameters and 650000 neurons
3. Five convolutional layers, some of which are followed by max-pooling layers.
4. three fully-connected layers with a final 1000-way softmax
5. non-saturating neuraons
6. a very efficient GPU implementation of the convolution
7. Using dropout
8. using ReLU nonlinearity is faster to training
9. Faster learning has a great influence on the performance of large models trained on large datasets.
10. Dropout roughly doubles the number of iterations required to converge
11. momentum and weight decay

#### 3. Visualizing and Understanding Convolutional Networks (zfnet)
1. Show why cnn perform so well and how to improve it.
2. introduce a visualization technique that reveal the input stimuli that excit individual feature map at any layer in the model.
3. observe the evolution of feature during training and diagnose potential problem with the model. 
4. multi-layer deconvolution network: project the feature activations back to the input pixel space.

