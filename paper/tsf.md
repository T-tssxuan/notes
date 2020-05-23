## 序列预测

### 1. Forecasting at Scale
- 把预测分解札总体趋势、周期变化、特定事件
- 使用周期序列模拟改变点

### 2. Time Series Forecasting With Deep Learning: A Survey
- 

### 3. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, 2018.4
- 使用CNN和RNN结合，关注短期和长期信息
- 成功的序列预测应该能够发现这其中的周期变化
- 添加skip-link
- 对结果进行分解，分为线性和非线性部分

### 4. Temporal Pattern Attention for Multivariate Time Series Forecasting, 2019.9
- 常规的rnn加attention可能会导到丢失时空模式
- AR和VAR模型都不能对非线性进行良好的拟合
- RNN FOR NLP一般一次只输入一个主变量
- 
