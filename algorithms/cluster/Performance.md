## Performance
- Adjusted Rand index¶
- Normalized Mutual Information(NMI) and Adjusted Mutual Information(AMI)
- homogeneity: each cluster contains only members of a single class.
- completeness: all members of a given class are assigned to the same cluster.
- homogeneity and completeness harmonic mean called V-measure
- The Fowlkes-Mallows index (sklearn.metrics.fowlkes_mallows_score) can be used when the ground truth class assignments of the samples is known
- Silhouette Coefficient: If the ground truth labels are not known, evaluation must be performed using the model itself.
- Calinski-Harabaz Index: If the ground truth labels are not known, the Calinski-Harabaz index can be used to evaluate the model, where a higher Calinski-Harabaz score relates to a model with better defined clusters.
