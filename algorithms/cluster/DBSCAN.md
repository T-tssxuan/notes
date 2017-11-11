## DBSCAN
1. The DBSCAN algorithm views clusters as areas of high density separated by areas of low density.
2. The clusters to which non-core samples are assigned can differ depending on the data order.
3. This implementation is by default not memory efficient because it constructs a full pairwise similarity matrix in the case where kd-trees or ball-trees cannot be used (e.g. with sparse matrices). This matrix will consume n^2 floats. 
