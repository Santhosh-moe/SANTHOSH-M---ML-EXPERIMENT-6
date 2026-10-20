## DATE: 06/10/2026





## AIM: 






Apply KMeans to a synthetic 2-cluster dataset; compute and visualize inertia. 






## ALGORITHM : 






Lloyd’s algorithm: assign to nearest centroid → recompute centroids until convergence. 





## THEORY:






KMeans minimizes within-cluster variance; relates to vector quantization. 


```


## INPUT: 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
np.random.seed(42) 
X = np.vstack([np.random.randn(50,2)+np.array([0,0]), 
np.random.randn(50,2)+np.array([5,5])]) 
kmeans = KMeans(n_clusters=2, random_state=42).fit(X) 
labels = kmeans.labels_ 
centers = kmeans.cluster_centers_ 
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis'); 
plt.scatter(centers[:,0], centers[:,1], marker='x', color='red', 
s=100); plt.title('KMeans Clusters'); plt.show() 
print('Inertia (WCSS):', round(kmeans.inertia_,3)) 
DETAILED EXPLANATION: 
1. Generate two Gaussian blobs for two clusters. 
2. KMeans(...).fit(X) runs the clustering iterations; labels_ are assignments; 
cluster_centers_ are learned centroids. 
3. Plot points colored by cluster and overlay centroids. 
4. inertia_ measures within-cluster sum-of-squares. 



## OUTPUT: 
Scatter plot showing two clusters; Inertia (WCSS): 180.342 (value varies).


```



## RESULT: 
Students observe cluster structure and effect of centroid positions. Note sensitivity to 
initialization & scaling. 



