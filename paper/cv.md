## CV related paper

### 1. Deep Residual Learning for Image Recognition(Residual Net)
1. 原始问题变成F(x) = H(x) - x
2. 网络复杂度低于VGG net
3. 有很高的泛化，可以用于视觉和非视觉
4. 使用y = F(x, {Wi}) + x没有添加额外的复杂度，也没有增加过多的计算量
5. F函数可以很灵活，两到三层都可以，但如果只有一层，可能不怎么适合

### 2. ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
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

### 3. Visualizing and Understanding Convolutional Networks (zfnet)
1. Show why cnn perform so well and how to improve it.
2. introduce a visualization technique that reveal the input stimuli that excit individual feature map at any layer in the model.
3. observe the evolution of feature during training and diagnose potential problem with the model. 
4. multi-layer deconvolution network: project the feature activations back to the input pixel space.

### 4. Semantic Soft Segmentation
1. 本文从分割的角度出发，用神经网络的结果表示图像的材质和颜色
2. 使用egendecomposition和laplacian matrix进行分割

### 5. Semantic Human Matting
1. 创建一个大数据集
2. SHM水平跟interactive matting methods相当
3. 使用两个网络进行，即T-Net和M-Net
4. RGB,trimap构成了6channel的输入

### 6. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- 同时扩展depth/width/resolution，达到更好的扩展效果
- 使用NAS搜索更好的基础结构

### 7. You Only Look Once: Unified, Real-Time Object Detection
- 不像传统网络，YOLO每次对整张图进行全局处理
- YOLO比Fast R-CNN少了一半的背景错误
- 主要设计
  - 把图像分成SXS个网格，每个网格预测B个box
  - 每个box预测相对所以网格的x, y, w, h, conf
  - 每个grid预测相应的类别
  - 最后每个图片就是预测：S X S X (B X 5 + C)
- 直接用squared error可以导致不均匀的问题，对包涵OBJ的告乘以5，对于不包涵OBJ的乘以0.5
- 为了处理大的box和小的box不一致的问题，直接预测box高和宽的平方根
- 每个predictor都专门化

### 8. YOLO9000: Better, Faster, Stronger
- 使用join train能够使模型
- 添加Batch Normalization
- 使用更高分辩率的分类器
- Convolutional With Anchor Boxes
- 使用dimension cluster
- Direct location prediction.
- 使用层级概念
- 使用wordnet联合不同的数据集

### 9. YOLOv3: An Incremental Improvement
- 使用lr可以应对多label数据
- 使有更多的feature map，包括up sampling，以及更前面的层

### 10. IMAGEBERT: CROSS-MODAL PRE-TRAINING WITH LARGE-SCALE WEAK-SUPERVISED IMAGE-TEXT DATA
- 上千万量级的图片和描述信息
- 图文bert

### 11. Bag of Tricks for Image Classification with Convolutional Neural Networks
- Large-batch training
  - 线性增加lr
  - lr warmup
  - 针对最后一个block的BN层的gamma置0
  - 只针对cnn和fc层进行weight decay，别的层不使用
- 低精度训练：使用fp16可以提速2到3倍，使用混合精度能同时提升训练速度和精度
- 使用不同的网络结构
- 使用consine learning rate decay
- 使用label smoothing
- 使用知识蒸馏

### 8. Rich feature hierarchies for accurate object detection and semantic segmentation

### 9. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

### 10. Object Detection with Discriminatively Trained Part Based Models
