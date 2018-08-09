## Reinforcement Learning related paper

### 1. Playing Atari with Deep Reinforcement Learning
1. Learning to control agents directly from high-dimensional sensory inputs like vision and speech is one of the long-standing challenges of reinforcement learning.
2. RL algorithms must be able to learn from a scalar reward signal that is frequently sparse, noisy and delayed. The delay between actions and resulting rewards, which can be thousands of timesteps long, seems particularly daunting when compared to the direct association between inputs and targets found in supervised learning.
3. This paper demonstrates that a convolutional neural network can overcome these challenges to learn successful control policies from raw video data in complex RL environments.
4. The goal of the agent is to interact with the emulator by selecting actions in a way that maximises future rewards.
5. The basic idea behind many reinforcement learning algorithms is to estimate the actionsvalue function.
6. Deep neural networks have been used to estimate the environment E; restricted Boltzmann machines have been used to estimate the value function or the policy.
7. The divergence issues with Q-learning have been partially addressed by gradient temporal-difference methods.

### 2. Mastering the Game of Go without Human Knowledge
1. AlphaGo 使用树算法，来衡量位置，以及移动的选择
2. 纯基于强化学习的算法，不基于任何人类引导，领域知识
3. AlphaGo 自己教自己
4. AlphaGo Fan&Lee policy network使用监督学习，预测人类步骤，并使用强化学习调整; value network用于预测游戏相互搏弈的胜者
5. AlphaGo Zero: 1. 完全无人类参与; 2. 只使用白子和黑子用做输入; 3. 使用单一网络，而非分开policy和value网络; 4. 只使用树搜索来衡量位置和样例移动，不使用MonteCarlo rollout
6. 一个新的reinforcement learning嵌入在训练过程中

### 3. Mastering the game of Go with deep neural networks and tree search
1. 搜索树的复杂度可以近似为b^d，b是每一步合法选择，d是游戏的长度
2. 降低复杂度的方法：1. 对树进行剪枝，然后进行近似回报预测，在chess, checker, othello中可用，但在围棋中不合适，2. 在状态s对行动进行采样
3. RL强于SL
4. Monte Carlo rollouts, fast rollout policy
5. PUCT, value network, 
6. 网络，输入，value, policy

### 4. Vector-based navigation using grid-like representations in artificial agents

### 5. End-to-End Reinforcement Learning for Automatic Taxonomy Induction
1. abandon the previous two stage method, because they believe the two stage style may suffer from error propagation.
2. The two stage stye method isolated the two stage of the classify, and the different level of edge does not equally.

### 6. DDPG algorithm

### 7. An MDP-based recommender system


### 8. Improving recommender systems with adaptive conversational strategies

### 9. Deep Reinforcement Learning with Attention for Slate Markov Decision Processes with High-Dimensional States and Actions, 2015

### 10. A hybrid web recommender system based on q-learning, 2008

### 11. Deep reinforcement learning in large discrete action spaces

### 12. Continuous control with deep reinforcement learning

### 6. Asynchronous Methods for Deep Reinforcement Learning

### 4. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
1.  
