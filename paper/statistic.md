## Statistic related paper

#### 1. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift 
1. 训练神经网络十分困难的原因在于每一层的分布随时间改变
2. batch normalization可以使我们在不那么精细的微调达到较高的学习率
3. 使用mini-batch是对全局的一种代替；一次训练多个，比一个个训练更加高效
4. 子网络的分布稳定，对其它网络的训练有增益效果
5. 通过规范化，可以稳定其均值和标准差
6. 通过处理输入，使用分布归一化，可以减弱bad case的影响
7. 
