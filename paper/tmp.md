#### Targeted Coupon Distribution Using Social Networks
1. 使用社交网络，提取用户消费力
2. 瞎发优惠券可能起反作用
3. 如何介入，如何控制比例
4. 使用队列以及反馈控制发券量
5. 


# The gan related pater

### 1. Stacked Generative Adversarial Networks
1. bottom-up模式一般都是专注于抽取有用的表征，而对数据的分布无能为力
2. 引入representation discriminators用于使得SGAN的中间表示保持在DNN的流形上
3. 除了adversarial loss，还引入了conditional loss用于使得生成网络依赖于上层输入，引入novel entropy loss使得生成样本足够分散
4. 相对于使用pre-train或者perceptual loss的方法，SGAN在中间生成表示的loss，而非只关注于最后的loss
5. 
