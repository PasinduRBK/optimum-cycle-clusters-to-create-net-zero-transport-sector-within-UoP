import numpy as np
from sklearn.cluster import MeanShift 
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
centers = [[1,1],[5,5],[7,9]]
X, _ = make_blobs(n_samples = 200,centers = centers,cluster_std = 1)

plt.scatter(X[:,0],X[:,1])
plt.show()
print(X)

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)

n_clusters_ = len(np.unique(labels))

print("Number of estimated clusters",n_clusters_)

colors = 10*('r.','g.','b.','c.','k.','y.','m.')

print(colors)
print(labels)

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize = 10)

plt.scatter(cluster_centers[:,0],cluster_centers[:,1],marker="x",s=150,linewidths = 5,zorder = 10)

plt.show()
