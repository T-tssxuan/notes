## CNN related paper

#### 1. Deep Residual Learning for Image Recognition(Residual Net)
1. 原始问题变成F(x) = H(x) - x
2. 网络复杂度低于VGG net
3. 有很高的泛化，可以用于视觉和非视觉
4. 使用y = F(x, {Wi}) + x没有添加额外的复杂度，也没有增加过多的计算量
5. F函数可以很灵活，两到三层都可以，但如果只有一层，可能不怎么适合
