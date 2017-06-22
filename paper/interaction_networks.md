## Interaction Networks Related

#### 1. VAIN: Attentional Multi-agent Predictive Modeling(思想可以，结果不行)
1. interaction networks针对multi-agent建模提出，在本文中，我们提出VAIN，一个multi-agent预测模型，基于attention架构
2. Interaction Networks每一步需要O(N^2)的计算量，CommNets每一步的计算，需要O(N)的计算量，而且没有明确定义交互，而所计算量全转到theta上了；本论文基于attention改进了前面两个网络的缺点
3. 每个agent都有一个特征输入，Es用于生成单个编码，Ec用于生成attention编码和communication编码
4. 在Chess Piece Prediction、Soccer Players、Bouncing Balls上进行训练，并对比了相关结果，基本上能赶到interaction networks，相参CommNets准确度有所提高
