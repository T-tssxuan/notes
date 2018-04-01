## Neural Network related paper

### 1. DyNet: The Dynamic Neural Network Toolkit
1. In the static decalaration strategy that is used in toolkits like Theano, CNTK, and TensorFlow, the user first defines a computation graph, and then examples are fed into an engine that executes this computation and computes its derivative.
2. One chanllenge with dynamic declaration is that because the symbolic computation graph is defined anew for every training example, its construction must have low overhead.
3. Rapid prototyping and easy maintenance of efficient and correct model code is of paramount importance in deep learning.
4. Since the differentiation of composite expressions is algorithmically straightforward, there is little downside to using autodiff algorithms in placeof hand-written code for computing derivatives.
5. Tow steps of static declaration: 1) Definition of a computational architecture; 2) Execution of the computation.
6. Advantages of static declaration: 1) Optimization. 2) Schedule computation resources. 3) Benefit to the toolkit designer.
7. Disadvantages of static declaration: 1) Variably sized inputs. 2) Variably structured inputs. 3) Nontrivial inference algorithms. 4) Variably structured outputs. 5) Difficulty in expressing complex flow-control logic. 6) Complexity of the computation graph impolementation. 7) Difficulty in debugging.
8. DyNet: there are no separate steps for definition and execution: the necessary computation graph is created, on the fly, as the loss calculation is executed, and a new graph is created for each training instance.
9. Advantageous of DyNet: 1) Define a different computation architecture for each training example of batch, allowing for the handling of variably sized or structured input using flow-control facilities of the host language. 2) Interleave definition and execution of computation, all  for the handling of cases where the structure of computation may change depending on the results of previous computation steps.

### 2. An overview of gradient descent optimization algorithms
1. 有三个gradient descent的变种，即：Batch gradient descent, Stochastic gradient descent, Mini-batch gradient descent，他们之间的主要区别在于用于计算梯度数据量；取决于数据量，我们在时间和精度之间进行权衡
2. Batch gradient descent: 计算所有数据后，进行一次更新；由于遍历所有数据才更新一次，效率很低；不能进行在线计算；可以保正在convex函数中找到最优解，和non-convex函数的次优解
3. Stochastic gradient descent: 在每一个样例的基础上进行参数更新；Batch gradient descent会进行很多冗余计算，如很多差不多的样本重复计算；sgd进行高频高方差的更新，会导致目标函数摆动相当严重；相对于bgd保证到当前最优，sgd可以跳到新的且可能更好的局部最优解；通过降低lr，可以防止overshooting，此时sgd更接近bgd，几乎能找到convex的全局最优解和non-covex的局部最优解
4. Mini-batch gradient descent: 降低参数更新方差，使收敛更稳定；可以利用常见机器学习库高度优化的矩阵计算功能；当见的50-256；
5. Mini-batch challenges: 不能保证好的收敛；选取好的lr很困难；在训练中对lr进行调整的常见方法，一般都需要相应的调度方法(schedule)和门槛(threshold)，但是提前设定的方法不一定适合一些特定的数据集；同样的lr被应用到所有的参数；如何处理局部次优解和saddle point问题
6. Momentum: 主要通过保持方向和减少摆动来加速训练；其使同一方向梯度得到保持并加速，方向变化大的梯度得到抑制，从而加速收敛，并减少震荡；v = gamma * v + lr * grad_param(theta), theta = theta - v
7. Nesterov accelerated gradient: 进行加速时，我们需要梯度能够在斜率下降时，相应的更新速度也下降，而Nesterov就是用于实现这一目标的；基本就是从theta中移除动量的影响，让我们提前预知参数的未来位置；v = gamma * v + lr * grad_param(theta - gamma * v), theta = theta - v；提前预知提升响应速度，其对RNN的性能提升很大
8. Adagrad: 自动为参数调节lr，在更新频次高的参数上进行小幅度更新，在更新频次低的参数上进行较大幅度更新；非常适合稀疏数据；Glove模型中使用Adagrad；theta = theta - lr * grad_param(t) / sqrt(Gt+ epsilon)；有意思的是，如果没有平方根，本算法表现相当差劲；不需要手动调节，一般设置为0.01；Adagrad的最大问题在于其分母是递增的，这样在一定程度后，参数将会得不到更新
9. Adadelta: 是一种解决Adagrad单调递减学习率的方法；相对于Adagrad保持所有递梯度信，Adadelta只保留最多w个最近的梯度信息，在实现时，使用历史值和当前梯度的插值来实现，最终可以变为RMS[g]；delta_theta = - RMS[delta_theta] * g / RMS[g], theta = theta + delta_theta；不需要设置学习率，初中期很快，后期摆动；
10. RMSprop: 本质上跟Adadelta相差不多；E[g^2] = 0.9 * E[g^2] + 0.1 * g^2, theta = theta - lr * g / (sqrt(E[g^2]) + epsilon)；Hinton lr=0.001
11. Adam: 引入梯度矩囝的一阶估计和二阶估计来调整参数学习率；其中一阶参数用于动量，二阶参数用于控制学习率；可以视为RMSprop 和 momentum的给合
12. AdaMax: Adam对于v实际使用的是l2正则，我们可以范化为lp正则；AdaMax使用的是linfinit 正则，就变成了u = max(beta * v, g)
13. Nadam: 

### 3. Neural Turing Machines
1. 结合神经网络，外部用于与注意力机制交互的资源。
2. 系统与图灵机、冯诺伊曼架构相信，但是其是端到端可微，并可以使用梯度下降进行训练
3. RNN是Turing-Complete，因此其具有模拟任何流程的能力
4. 本文从心理学，语言学，神经科学，以及人工智能和神经网络等主题展开
5. NTM主要有网络控制器，和记忆存储
6. 不像常规图灵机只操作一个内存单元，NTM使用attention机制与内存进行模糊读写操作
7. content-based addressing: 基于控制器值和位置内容值之间的相似程度（简单）; 
cation-based addressing: 基于位置的寻址方式
8. 寻址系统由三部分组成：基于内容; 基于内容的结果可以选择和偏移，使得可以进行连续寻址; 前一步的结果，可以用于迭代寻址，访问间隔地址序列
9. 使用循环神经网络作为网络控制器


### 4. Group Normalization
1. a replace for the batch normalization
2. the drawback of the batch normalization: batch not always exist, variance change, batch should be bigger.

