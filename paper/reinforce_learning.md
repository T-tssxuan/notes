## Reinforcement Learning related paper

### Playing Atari with Deep Reinforcement Learning
- Learning to control agents directly from high-dimensional sensory inputs like vision and speech is one of the long-standing challenges of reinforcement learning.
- RL algorithms must be able to learn from a scalar reward signal that is frequently sparse, noisy and delayed. The delay between actions and resulting rewards, which can be thousands of timesteps long, seems particularly daunting when compared to the direct association between inputs and targets found in supervised learning.
- This paper demonstrates that a convolutional neural network can overcome these challenges to learn successful control policies from raw video data in complex RL environments.
- The goal of the agent is to interact with the emulator by selecting actions in a way that maximises future rewards.
- The basic idea behind many reinforcement learning algorithms is to estimate the actionsvalue function.
- Deep neural networks have been used to estimate the environment E; restricted Boltzmann machines have been used to estimate the value function or the policy.
- The divergence issues with Q-learning have been partially addressed by gradient temporal-difference methods.

### Mastering the Game of Go without Human Knowledge
- AlphaGo 使用树算法，来衡量位置，以及移动的选择
- 纯基于强化学习的算法，不基于任何人类引导，领域知识
- AlphaGo 自己教自己
- AlphaGo Fan&Lee policy network使用监督学习，预测人类步骤，并使用强化学习调整; value network用于预测游戏相互搏弈的胜者
- AlphaGo Zero: 1. 完全无人类参与; 2. 只使用白子和黑子用做输入; 3. 使用单一网络，而非分开policy和value网络; 4. 只使用树搜索来衡量位置和样例移动，不使用MonteCarlo rollout
- 一个新的reinforcement learning嵌入在训练过程中

### Mastering the game of Go with deep neural networks and tree search
- 搜索树的复杂度可以近似为b^d，b是每一步合法选择，d是游戏的长度
- 降低复杂度的方法：1. 对树进行剪枝，然后进行近似回报预测，在chess, checker, othello中可用，但在围棋中不合适，2. 在状态s对行动进行采样
- RL强于SL
- Monte Carlo rollouts, fast rollout policy
- PUCT, value network, 
- 网络，输入，value, policy


### End-to-End Reinforcement Learning for Automatic Taxonomy Induction
- abandon the previous two stage method, because they believe the two stage style may suffer from error propagation.
- The two stage stye method isolated the two stage of the classify, and the different level of edge does not equally.

### The Arcade Learning Environment: An Evaluation Platform for General Agents, 2013
- 一个对Arcade game的普适化软件
- 验证了几种RL方法

### Human-level control through deep reinforcement learning, 2015
- deep Q-network end2end
- Q的一点点改变，可能导致很大的结果变化
- 引入experience replay
- 引入periodically updated

### Rainbow: Combining Improvements in Deep Reinforcement Learning, 2018
- 本文测试了DQN的6种扩展，并对他们的结合进行细致的研究，结果证明，组合结果无论在效率还是性能上都达到了当前最优的结果。
- 分开online network和target network可以提升agent的性能
- 本文选取了6个扩展：Double Q-learning, Prioritized replay, Dueling networks, Multi-step learning, Distributional RL, Noisy Nets

### PRIORITIZED EXPERIENCE REPLAY, 2016
- 本文的关键点在于：RL agent可以从一些经验中更有效的学习
- TD error provide one way to measure these priorities
- 需要处理好噪声、发散性等问题
- 结合优先队列和均匀分布两个进行
- prioritized replay会导致分布向一个不可变的方向变化，因此需要进行控制，本部分引入importance-sampling weights

### Asynchronous Methods for Deep Reinforcement Learning, 2016
- 介绍一种异少强化学习训练的方法，在4种算法上得到验证
- 提出A3C算法
- 在多个线程上运行多个actor，他们的经验的相关性可能更少，因此不使用依赖experience replay的DQN来保证算法的稳定性
- 使用多个agent增加了探索的发散性

### Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents, 2018
- multiple different tasks, interesting and challenging for humans, free
- 1) 一些关于ALE的经验，2) 新的ALE版本，3) 新的benchmark，和reproducibility experiment，4) 总结最新结果，以及当前待解决的问题，5) 介绍平台的特性
- 规范化的分数可以更容易找到独立于游戏的算法: 1) Frame skipping, 2) Color averaging and frame pooling, 3) frame stacking.
- 建议只有游戏结束才算是终结状态
- sticky actions

### Deep Reinforcement Learning with Double Q-learning, 2015
- 本文主要回答了关于DQN overestimate相关的问题
- 由于最大化操作，Q-learning有时会学习到一些大得不真实的action-value
- 证明Double DQN不可得到更高的分数，也证明了overestimate对结果有害
- 常规的DQN使用同一参数选择和评估，这很容易导致过于乐观的估计
- Double Q-learning分解了动作选择和评估
- overestimate可能会更加的恶化，当值已经过于乐观化时
- 有些情况overoptimism不一定是对结果有害，但是减少overoptimism可以增加稳定性

### Simulation and Visual- ization of Multi-Joint Dynamics with Contact


### Massively Parallel Methods for Deep Reinforcement Learning, 2018

### Actor-Critic Algorithms, 2000


### Weighted importance sampling for off-policy learning with linear function approximation

### Deep Reinforcement Learning with Double Q-learning

### Vector-based navigation using grid-like representations in artificial agents

### 6. DDPG algorithm

### 7. An MDP-based recommender system


### 8. Improving recommender systems with adaptive conversational strategies

### 9. Deep Reinforcement Learning with Attention for Slate Markov Decision Processes with High-Dimensional States and Actions, 2015

### 10. A hybrid web recommender system based on q-learning, 2008

### 11. Deep reinforcement learning in large discrete action spaces

### 12. Continuous control with deep reinforcement learning

### 6. Asynchronous Methods for Deep Reinforcement Learning

### 4. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
-  

### 5. Policy gradient methods for reinforcement learning with function approximation

### 6. Deterministic policy gradient algorithms.

### 7. Near-optimal reinforcement learning in polynomial time

### 8. Learning from delayed rewards

### 9. Continuous control with deep reinforcement learning
