from __future__ import division
from src import SIWO, community, utils
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from time import time
from matplotlib import pyplot as plt
import matplotlib.cm as cmap
import networkx as nx
import numpy as np


# global variable for color maps
colormap = cmap.tab10.colors

# Read the graph
graph_paths = ['../data/toy_0.graph', '../data/toy_1.graph', '../data/toy_2.graph', '../data/toy_3.graph']
graphs_original = []  # store original networkx graph objects
graphs_redist = []  # store graphs with redistributed node attribute
partitions_gt = []  # store ground-truth partitions generated by DANCer
for g_path in graph_paths:
    graph, partition = utils.read_dancer_data(g_path)
    graphs_original.append(graph)
    partitions_gt.append(partition)

'''
Experiment 1: well-separated clusters
'''
# attribute redistribution
for i in range(4):
    g = graphs_original[i]
    partition = partitions_gt[i]
    graphs_redist.append(utils.redistribute_attribute(g, partition, 0.5, 19910420))

print("within inertia after redistribution:")
for (g, p) in zip(graphs_redist, partitions_gt):
    within_inertia = utils.compute_within_inertia(g, p)
    print(within_inertia)

# louvain
c_louvain = []
print("Louvain ...")
for (graph, partition) in zip(graphs_redist, partitions_gt):
    start = time()
    louvain_comm = community.best_partition(graph)
    end = time()
    c_louvain.append(louvain_comm)
    print('nmi, ari')
    print(normalized_mutual_info_score(louvain_comm.values(), partition.values()))
    print(adjusted_rand_score(louvain_comm.values(), partition.values()))
    print('time: {}'.format(end - start))

# siwo
c_siwo = []
print("SIWO ...")
for (graph, partition) in zip(graphs_redist, partitions_gt):
    start = time()
    siwo_comm = SIWO.best_partition(graph)
    end = time()
    c_siwo.append(siwo_comm)
    print('nmi, ari')
    print(normalized_mutual_info_score(siwo_comm.values(), partition.values()))
    print(adjusted_rand_score(siwo_comm.values(), partition.values()))
    print('time: {}'.format(end - start))

# spectral_clustering
true_clusters = [10, 10, 10, 10]
c_spectral = []
print("only sc ...")
for (graph, k, partition) in zip(graphs_redist, true_clusters, partitions_gt):
    start = time()
    X = nx.get_node_attributes(graph, 'attr').values()
    D = euclidean_distances(X, X)
    Sim = np.exp(-D)
    sc = SpectralClustering(k, affinity='precomputed', n_init=20, assign_labels='discretize')
    sc_clusters = sc.fit_predict(Sim)
    end = time()
    print('nmi, ari')
    print(normalized_mutual_info_score(sc_clusters, partition.values()))
    print(adjusted_rand_score(sc_clusters, partition.values()))
    print('time: {}'.format(end - start))


print('only dbscan')
c_db = []
for (graph, k, partition) in zip(graphs_redist, true_clusters, partitions_gt):
    start = time()
    X = nx.get_node_attributes(graph, 'attr').values()
    D = euclidean_distances(X, X)
    avg_degree = np.mean(graph.degree().values())
    db = DBSCAN(min_samples=avg_degree)
    db_clst = db.fit_predict(X)
    end = time()
    print("nmi, ari")
    print(normalized_mutual_info_score(db_clst, partition.values()))
    print(adjusted_rand_score(db_clst, partition.values()))
    print('time: {}'.format(end - start))

#
# # spectral clustering + louvain
# print("SC + louvain ...")
# c_sc = []
# for (graph, k, partition) in zip(graphs_redist, true_clusters, partitions_gt):
#     start = time()
#     X = nx.get_node_attributes(graph, 'attr').values()
#     D = euclidean_distances(X, X)
#     Sim = np.exp(-D)
#     sc = SpectralClustering(k, affinity='precomputed', n_init=20, assign_labels='discretize')
#     sc_clusters = sc.fit_predict(Sim)
#     louvain_comm = community.best_partition(graph)
#     graph_integrated = utils.naive_combine(sc_clusters, louvain_comm.values(), 0.5)
#     part_louvain = community.best_partition(graph_integrated)
#     end = time()
#     print("nmi, ari")
#     print(normalized_mutual_info_score(part_louvain.values(), partition.values()))
#     print(adjusted_rand_score(part_louvain.values(), partition.values()))
#     print('time: {}'.format(end - start))
#
#
# # DBSCAN + louvain
# print("DBSCAN + louvain ...")
# c_db = []
# for (graph, k, partition) in zip(graphs_redist, true_clusters, partitions_gt):
#     start = time()
#     X = nx.get_node_attributes(graph, 'attr').values()
#     D = euclidean_distances(X, X)
#     avg_degree = np.mean(graph.degree().values())
#     db = DBSCAN(min_samples=avg_degree)
#     db_clst = db.fit_predict(X)
#     louvain_comm = community.best_partition(graph)
#     graph_integrated = utils.naive_combine(db_clst, louvain_comm.values(), 0.5)
#     part_louvain = community.best_partition(graph_integrated)
#     end = time()
#     print("nmi, ari")
#     print(normalized_mutual_info_score(part_louvain.values(), partition.values()))
#     print(adjusted_rand_score(part_louvain.values(), partition.values()))
#     print('time: {}'.format(end - start))
#
# # siwo + sc
# print("siwo + sc ...")
# graphs_siwo_sc = []
# for (graph, k, partition) in zip(graphs_redist, true_clusters, partitions_gt):
#     start = time()
#     X = nx.get_node_attributes(graph, 'attr').values()
#     D = euclidean_distances(X, X)
#     Sim = np.exp(-D)
#     sc = SpectralClustering(k, affinity='precomputed', n_init=20, assign_labels='discretize')
#     sc_clusters = sc.fit_predict(Sim)
#     siwo_comm = SIWO.best_partition(graph)
#     graph_integrated = utils.naive_combine(sc_clusters, siwo_comm.values(), 0.5)
#     part_siwo = community.best_partition(graph_integrated)
#     end = time()
#     print("nmi, ari")
#     print(normalized_mutual_info_score(part_siwo.values(), partition.values()))
#     print(adjusted_rand_score(part_siwo.values(), partition.values()))
#     print('time: {}'.format(end - start))
#
#
# # siwo + dbscan
# print("siwo + db ...")
# graphs_siwo_db = []
# for (graph, k, partition) in zip(graphs_redist, true_clusters, partitions_gt):
#     start = time()
#     X = nx.get_node_attributes(graph, 'attr').values()
#     D = euclidean_distances(X, X)
#     avg_degree = np.mean(graph.degree().values())
#     db = DBSCAN(min_samples=avg_degree)
#     db_clst = db.fit_predict(X)
#     siwo_comm = community.best_partition(graph)
#     graph_integrated = utils.naive_combine(db_clst, siwo_comm.values(), 0.5)
#     part_siwo = community.best_partition(graph_integrated)
#     end = time()
#     print("nmi, ari")
#     print(normalized_mutual_info_score(part_siwo.values(), partition.values()))
#     print(adjusted_rand_score(part_siwo.values(), partition.values()))
#     print('time: {}'.format(end - start))
