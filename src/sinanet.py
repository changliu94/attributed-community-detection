from __future__ import division
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from src import SIWO, community, utils
import networkx as nx
import numpy as np
from src.IL import ML2
from time import time

data_path = '../data/sinanet/'

g, gt_partition = utils.read_sina_data(data_path)
within_inertia = utils.compute_within_inertia(g, gt_partition)
print("within inertia ratio: {}".format(within_inertia))

mod = community.modularity(gt_partition, g)
print("modularity: {}".format(mod))




'''
Louvain
'''
print("louvain")
start = time()
partition_louvain = community.best_partition(g)
end = time()
print('time: {}'.format(end - start))
print(normalized_mutual_info_score(gt_partition.values(), partition_louvain.values()))
print(adjusted_rand_score(gt_partition.values(), partition_louvain.values()))
print(len(set(partition_louvain.values())) / len(set(gt_partition.values())))

'''
SIWO
'''
print("siwo")
start = time()
partition_siwo = SIWO.best_partition(g)
end = time()
print('time: {}'.format(end - start))
print(normalized_mutual_info_score(gt_partition.values(), partition_siwo.values()))
print(adjusted_rand_score(gt_partition.values(), partition_siwo.values()))
print(len(set(partition_siwo.values())) / len(set(gt_partition.values())))

'''
sc
'''
print("sc")
k = 10
start = time()
X = nx.get_node_attributes(g, 'attr').values()
D = euclidean_distances(X, X)
Sim = np.exp(-D)
sc = SpectralClustering(k, affinity='precomputed', n_init=20, assign_labels='discretize')
sc_clusters = sc.fit_predict(Sim)
end = time()
print(normalized_mutual_info_score(sc_clusters, gt_partition.values()))
print(adjusted_rand_score(sc_clusters, gt_partition.values()))
print(len(set(sc_clusters)) / len(set(gt_partition.values())))
print("time: {}".format(end - start))

'''
kmeans
'''
print('kmeans')
start = time()
X = nx.get_node_attributes(g, 'attr').values()
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
kmeans_clusters = kmeans.labels_
end = time()
print("time: {}".format(end - start))
print(normalized_mutual_info_score(kmeans_clusters, gt_partition.values()))
print(adjusted_rand_score(kmeans_clusters, gt_partition.values()))
print(len(set(kmeans_clusters)) / len(set(gt_partition.values())))


'''
louvain + sc
'''
print('louvain + sc')
start = time()
partition_louvain = community.best_partition(g)
X = nx.get_node_attributes(g, 'attr').values()
D = euclidean_distances(X, X)
Sim = np.exp(-D)
sc = SpectralClustering(k, affinity='precomputed', n_init=20, assign_labels='discretize')
sc_clusters = sc.fit_predict(Sim)
g_integrated = utils.naive_combine(sc_clusters, partition_louvain.values(), 0.5)
partition_lf = community.best_partition(g_integrated)
end = time()
print('time: {}'.format(end - start))
print(normalized_mutual_info_score(partition_lf.values(), gt_partition.values()))
print(adjusted_rand_score(partition_lf.values(), gt_partition.values()))
print(len(set(partition_lf.values())) / len(set(gt_partition.values())))


'''
louvain + kmeans
'''
print('louvain + kemans')
start = time()
partition_louvain = community.best_partition(g)
X = nx.get_node_attributes(g, 'attr').values()
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
kmeans_clusters = kmeans.labels_
g_integrated = utils.naive_combine(kmeans_clusters, partition_louvain.values(), 0.5)
partition_lf = community.best_partition(g_integrated)
end = time()
print('time: {}'.format(end - start))
print(normalized_mutual_info_score(partition_lf.values(), gt_partition.values()))
print(adjusted_rand_score(partition_lf.values(), gt_partition.values()))
print(len(set(partition_lf.values())) / len(set(gt_partition.values())))


print('siwo + sc')
start = time()
partition_louvain = SIWO.best_partition(g)
X = nx.get_node_attributes(g, 'attr').values()
D = euclidean_distances(X, X)
Sim = np.exp(-D)
sc = SpectralClustering(k, affinity='precomputed', n_init=20, assign_labels='discretize')
sc_clusters = sc.fit_predict(Sim)
g_integrated = utils.naive_combine(sc_clusters, partition_louvain.values(), 0.5)
partition_lf = community.best_partition(g_integrated)
end = time()
print('time: {}'.format(end - start))
print(normalized_mutual_info_score(partition_lf.values(), gt_partition.values()))
print(adjusted_rand_score(partition_lf.values(), gt_partition.values()))
print(len(set(partition_lf.values())) / len(set(gt_partition.values())))


'''
siwo + kmeans
'''
print('siwo + kemans')
start = time()
partition_louvain = SIWO.best_partition(g)
X = nx.get_node_attributes(g, 'attr').values()
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
kmeans_clusters = kmeans.labels_
g_integrated = utils.naive_combine(kmeans_clusters, partition_louvain.values(), 0.5)
partition_lf = community.best_partition(g_integrated)
end = time()
print('time: {}'.format(end - start))
print(normalized_mutual_info_score(partition_lf.values(), gt_partition.values()))
print(adjusted_rand_score(partition_lf.values(), gt_partition.values()))
print(len(set(partition_lf.values())) / len(set(gt_partition.values())))

#
# '''
# ILouvain
# '''
# print('ilouvain')
# start = time()
# attributes_IL, author_index = utils.fit_IL(g)
# IL = ML2(g, attributes_IL, author_index)
# partition_il = IL.findPartition()
# end = time()
# print('time: {}'.format(end - start))
# print(normalized_mutual_info_score(partition_il.values(), gt_partition.values()))
# print(adjusted_rand_score(partition_il.values(), gt_partition.values()))
# print(len(set(partition_il.values())) / len(set(gt_partition.values())))
