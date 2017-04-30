## Neural Network related paper

#### 1. DyNet: The Dynamic Neural Network Toolkit
1. In the static decalaration strategy that is used in toolkits like Theano, CNTK, and TensorFlow, the user first defines a computation graph, and then examples are fed into an engine that executes this computation and computes its derivative.
2. One chanllenge with dynamic declaration is that because the symbolic computation graph is defined anew for every training example, its construction must have low overhead.
3. Rapid prototyping and easy maintenance of efficient and correct model code is of paramount importance in deep learning.
4. Since the differentiation of composite expressions is algorithmically straightforward, there is little downside to using autodiff algorithms in placeof hand-written code for computing derivatives.
5. Tow steps of static declaration: 1) Definition of a computational architecture; 2) Execution of the computation.
6. Advantages of static declaration: 1) Optimization. 2) Schedule computation resources. 3) Benefit to the toolkit designer.
7. Disadvantages of static declaration: 1) Variably sized inputs. 2) Variably structured inputs. 3) Nontrivial inference algorithms. 4) Variably structured outputs. 5) Difficulty in expressing complex flow-control logic. 6) Complexity of the computation graph impolementation. 7) Difficulty in debugging.
8. DyNet: there are no separate steps for definition and execution: the necessary computation graph is created, on the fly, as the loss calculation is executed, and a new graph is created for each training instance.
9. Advantageous of DyNet: 1) Define a different computation architecture for each training example of batch, allowing for the handling of variably sized or structured input using flow-control facilities of the host language. 2) Interleave definition and execution of computation, all  for the handling of cases where the structure of computation may change depending on the results of previous computation steps.
