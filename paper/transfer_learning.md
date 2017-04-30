## Transfer Learning related paper

### 1. A Survey on Transfer Learning
1. A major assumption in many machine learning and data mining algorithms is that the training and future data must be in the same feature space and have the same distribution. However, in many real-world applications, this assumption may not hold.
2. This survey focus on categorizing and reviewing the current progress on transfer learning for classification, regression and clustering problems.
3. Transfer learning allows the domains, tasks, and distributions used in training and testing to be different.
4. The fundamental motivation for Transfer learning in the field of machine learning was discussed in a NIPS-95 workshop on "Learning to Learn", which focused on the need for lifelong machine-learning.
5. Research on transfer learning has attracted more and more attention since 1995 in different names: learning to learn, life-long learning, knowledge transfer, inductive transfer, multi-task learning, knowledge consolidation, context-sensitive learning, knowledge-based inductive bias, meta learning, and incremental/cumulative learning.
6. The ability of a system to recognize and apply knowledge and skills learned in previous tasks to novel tasks.
7. In contrast to multi-tasks learning, rather than learning all of the source and target tasks simultaneously, transfer learning cares most about the target task.
8. The roles of the source and target task are no longer symmetric in transfer learning.
9. Definition: (Transfer learning) Given a source domain Ds and learning task Ts, a target domain Dt and learning task Tt, transfer learning aims to help improve the learning of the target predictive function ft(.) in Dt using the knowledge in Ds and Ts, while Ds != Dt or Ts != Tt.
10. Research issues: (1) What to transfer, (2) How to transfer, (3) When to transfer.
11. **negative transfer**: A transfer hurt the performance of learning in the target domain.
12. **Inductive transfer learning**: the target task is different from the source task, no matter when source and target domains are the same or not. 
13. **Transductive transfer learning**: the source and target tasks are the same, while the source and target domains are different.
14. **Unsupervised transfer learning**: similar to inductive transfer learning setting, the target task is different from but related to the source task. However, the unsupervised transfer learning focus on solving unsupervised learning tasks in the target domanin. There are no labeled data available in both source and target domains in traning.
15. TrAdaBoost: IAssumes that the source and target domain data use exactly the same set of features and labels, but the distributions of the data in the two domains are different. It attempts to iteratively re-weight the source domain data to reduce the effect of the bad source data while encourage the "good" source data to contribute more for the target domain.

