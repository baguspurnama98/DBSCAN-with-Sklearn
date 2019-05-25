import numpy as np

data = [[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]]

# Build DBSCAN model
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=3.5, min_samples=2, metric='euclidean')
model = model.fit(data)
labels = model.labels_
core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

#jumlah data
unique, counts = np.unique(labels, return_counts=True)
banyakData = list(counts)
dataUnik = list(unique)
for i in dataUnik :
    if i == -1 :
        print(str('\nOutlier/noise : ') +str(banyakData[0]))
    else :
        print(str('Banyak Data Pada Cluster ') + str(i+1) +str(' :')+str(banyakData[i]))

from sklearn import metrics
silhouette_scr = metrics.silhouette_score(data, labels)
print("Silhouette Coefficient: %0.3f" % silhouette_scr)


import matplotlib.pyplot as plt
unique_labels = set(labels)
colors = ['k','tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:cyan', 'y', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:purple']

for i, d in enumerate(data):
    col = colors[labels[i] + 1]
    plt.plot(d[0], d[1], '.', markerfacecolor=col,markeredgecolor='w', markersize=15)
plt.show()
