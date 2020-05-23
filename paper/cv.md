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

### 12. UniViLM: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation
- 不但使用不同的视频间进行采样，也在同一个视频内采样

### 13. Deep Layer Aggregation, 2019.01
- 引入两种layer aggregation: iterative deep aggregation, hierarchical deep aggregation

### 14. Momentum Contrast for Unsupervised Visual Representation Learning
- contrast loss可以用于定义目标之间的区别，达到表征学习的目的
- 直接copy query的encoder到key encoder中，导到失败的原因是encoder改变太快
- 解决key encoder失败的方法是使用momentum的更新
- 使用BN可能导致信息泄露，而导致效果不好

### 15. The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification
- 常见的方法
  - 子网络执行部分检测
  - 特征学习，最大化差异度
- 使用一个loss完成差异化特征学习和结构定位
- 使用不同的通道进行分类专门化
- 不同的通道class的差异化

### 16. Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification from the Bottom Up, 2019.03
- 使用Mask R-CNN进行对象检测，以及使用CRF-based segmentation进行instance分割
- 图像分类会陷入对最大不同的区分
- 使用CAM产生表征，使用CRF进行分割，再用Mask R-CNN进行分割，在此基础之上，再使用Mask R-CNN替换CAM特征，迭代几轮，达到目的

### 17. Learning Deep Features for Discriminative Localization
- 使用GAP可以使网络识别完整的对象
- GAP相对GMP可以使整个感知点受到训练

### 18. Exploring the Limits of Weakly Supervised Pretraining
- 使用单个类二分类，效果相当的差
- 使用softmax，且每个tag使用1/k的量
- 数据量提升和标签提升都可以有效的提升分类效果
- hashtag数量分布呈Zipfian分布，需要进行采样，主要是对于较少的图片进行上采样

### 19. Deep learning for fine-grained image analysis: A survey
- fine-grained image recognition, fine-grained image retrieval and fine-grained image generation.
- 高度相似子类之间的类间较小差异，组内姿态、尺度、旋转带来的较大差异
- 识别：1) 引入图像本身信息，2)引入web文本，描述等
- 常规的图像检索着重检测相似，细粒度检索着重于检测子类型

### 20. Learning Attentive Pairwise Interaction for Fine-Grained Classification
- 模仿人类，对于成对对比图像
- 可以快速的在其它的网络中进行插拔
- 同时使用rank loss和cross entropy

### 21. Self-training with Noisy Student improves ImageNet classification
- 1) 老师模型在标注数据上训练, 2) 学生模型在老师模型的结果上运行，3) 学生变老师继续进行迭代
- 和模型蒸馏的主要区别在于，本文使用跟teach差不多大的学生模型，并且添加噪声
- 几个总结
  - 使用大的teacher效果更好
  - 使用更多的无标注数据，效果好
  - soft label效果好
  - 数据平衡很重要
  - 标注和无标注一起训练的数果好于先用无标签预训
  - 使用大的比量有标签和无标签数据
  - 从零开始训练student模型效果更好

### 22. Designing Network Design Spaces
- 在移动设备上高效
- 提升了resnet
- 在GPU上比efficientnet快5倍
- 使用更的操作符可能达到更好的效果

### 22. Fisher Kernels on Visual Vocabularies for Image Categorization


### 20. RPC: A Large-Scale Retail Product Checkout Dataset, 19.01
- 
 

### 8. Rich feature hierarchies for accurate object detection and semantic segmentation

### 9. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

### 10. Object Detection with Discriminatively Trained Part Based Models
