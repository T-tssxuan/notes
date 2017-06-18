## Statistic related paper

#### 1. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift 
1. 训练神经网络十分困难的原因在于每一层的分布随时间改变
2. batch normalization可以使我们在不那么精细的微调达到较高的学习率
3. 使用mini-batch是对全局的一种代替；一次训练多个，比一个个训练更加高效
4. 子网络的分布稳定，对其它网络的训练有增益效果
5. 通过规范化，可以稳定其均值和标准差
6. 通过处理输入，使用分布归一化，可以减弱bad case的影响

#### 2. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
1. 提出一种新的评估方法，主要思想是使用非线性logistic回归区分观察到的结果和生成的结果
2. 一个片理normalization的方法是，忽略Z(alpha)，把其视为常量
3. 通过对比数据和噪声，来学习数据中的一些特性，我们把这称为：通过对比学习
4. Pn(.)需要保证为正，这样才能保证Pd能够找到负例，然后基于负例进行推断
5. 由于normalized是做为估算的c常量，这样在目标达到最大时，也会使得alpha, c达到最优
6. 关于对比噪声的分布的选择: a. 容易生成，b. 能够通过log-pdf进行分析，c. 可以推导出较小的MSE，可选的有：高斯、混合高斯、ICA分布
7. 一般来说，噪声分布应该接近数据分布，否则可能学不到有用的信息
