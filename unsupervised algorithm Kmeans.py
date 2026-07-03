import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X,_=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=0)
kmeans=KMeans(n_clusters=4,random_state=0)
kmeans.fit(X)
centers=kmeans.cluster_centers_
labels=kmeans.labels_
plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis')
plt.scatter(centers[:,0],centers[:,1],c='red',marker='x',s=200,label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()
