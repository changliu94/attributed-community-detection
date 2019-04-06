from __future__ import division
import networkx as nx
import numpy as np
from time import time
from src import SIWO, community, utils
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
from collections import Counter

fb_path = '../data/facebook/facebook/'
network_id = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
graph_prefixes = [fb_path + str(i) for i in network_id]

# Create graphs and ground truth
graphs = []
partitions_gt = []
for gp in graph_prefixes:
    graph, partition = utils.read_fb_data(gp)
    partition = utils.add_outliers(graph, partition)
    print("num of real edges: {}".format(len(graph.edges())))
    graphs.append(graph)
    partitions_gt.append(partition)


# Get graph property
for (part, g) in zip(partitions_gt, graphs):
    mod = community.modularity(part, g)
    # print(mod)

sim_mats = []
for graph in graphs:
    X = utils.get_attr_mat(graph)
    sim = np.matmul(X, X.T)
    sim_mats.append(sim)


'''
Median threshold weighted graph
'''
graphs_attr_median = []
for sim in sim_mats:
    triu_inds = np.triu_indices(len(sim), 1)
    sim_vec = sim[triu_inds]
    threshold = np.median(sim_vec)
    sim[sim <= threshold] = 0
    np.fill_diagonal(sim, 0)
    graph = nx.from_numpy_matrix(sim)
    graphs_attr_median.append(graph)

# louvain
print("Median threshold weighted graph:")
c_virtual_median = []
for g, p_gt in zip(graphs_attr_median, partitions_gt):
    p = community.best_partition(g)
    c_virtual_median.append(p)
    print("nmi, ari, ratio")
    print(normalized_mutual_info_score(p_gt.values(), p.values()))
    print(adjusted_rand_score(p_gt.values(), p.values()))
    print(len(set(p.values())) / len(set(p_gt.values())))


'''
Same-edge threshold unweighted
'''
graphs_attr_same_edge = []
for sim, g in zip(sim_mats, graphs):
    m = len(g.edges())  # number of edges
    n = len(g.nodes())  # number of nodes
    total_possible_edges = n * (n-1) / 2
    ratio = m / total_possible_edges
    quantile = 1 - ratio

    triu_inds = np.triu_indices(len(sim), 1)
    sim_vec = sim[triu_inds]
    threshold = np.quantile(sim_vec, quantile)
    sim[sim <= threshold] = 0
    sim[sim > threshold] = 1
    np.fill_diagonal(sim, 0)
    graph = nx.from_numpy_matrix(sim)
    graphs_attr_same_edge.append(graph)

# louvain
print("same-edge threshold unweighted graph:")
c_virtual_same_edge = []
for g, p_gt in zip(graphs_attr_same_edge, partitions_gt):
    p = community.best_partition(g)
    c_virtual_same_edge.append(p)
    print("nmi, ari, ratio")
    print(normalized_mutual_info_score(p_gt.values(), p.values()))
    print(adjusted_rand_score(p_gt.values(), p.values()))
    print(len(set(p.values())) / len(set(p_gt.values())))

'''
Louvain
'''
c_louvain = []
print('Only Louvain ...')
for (graph, partition) in zip(graphs, partitions_gt):
    louvain_comm = community.best_partition(graph)
    c_louvain.append(louvain_comm)
    print("nmi, ari, ratio")
    print(normalized_mutual_info_score(louvain_comm.values(), partition.values()))
    print(adjusted_rand_score(louvain_comm.values(), partition.values()))
    print(len(set(louvain_comm.values())) / len(set(partition.values())))


'''
SIWO
'''
c_siwo = []
print('Only SIWO ...')
for (graph, partition) in zip(graphs, partitions_gt):
    louvain_comm = SIWO.best_partition(graph)
    c_siwo.append(louvain_comm)
    print("nmi, ari, ratio")
    print(normalized_mutual_info_score(louvain_comm.values(), partition.values()))
    print(adjusted_rand_score(louvain_comm.values(), partition.values()))
    print(len(set(louvain_comm.values())) / len(set(partition.values())))


'''
Louvain + same-edge unweighted
'''
print("Louvain + same-edge unweighted")
# create integrated graph
graphs_louvain_se = []
for (p_s, p_a) in zip(c_louvain, c_virtual_same_edge):
    c_attr = p_a.values()
    c_struct = p_s.values()
    graph_integrated = utils.naive_combine(c_attr, c_struct, 0.5)
    graphs_louvain_se.append(graph_integrated)

# use Louvain to cluster
for (g, part) in zip(graphs_louvain_se, partitions_gt):
    part_louvain = community.best_partition(g)
    print("nmi, ari, ratio")
    print(normalized_mutual_info_score(part_louvain.values(), part.values()))
    print(adjusted_rand_score(part_louvain.values(), part.values()))
    print(len(set(part_louvain.values())) / len(set(part.values())))


'''
Louvain + median weighted
'''
print("Louvain + median weighted")
# create integrated graph
graphs_louvain_median = []
for (p_s, p_a) in zip(c_louvain, c_virtual_median):
    c_attr = p_a.values()
    c_struct = p_s.values()
    graph_integrated = utils.naive_combine(c_attr, c_struct, 0.5)
    graphs_louvain_median.append(graph_integrated)

# use Louvain to cluster
for (g, part) in zip(graphs_louvain_median, partitions_gt):
    part_louvain = community.best_partition(g)
    print("nmi, ari, ratio")
    print(normalized_mutual_info_score(part_louvain.values(), part.values()))
    print(adjusted_rand_score(part_louvain.values(), part.values()))
    print(len(set(part_louvain.values())) / len(set(part.values())))


'''
siwo + same-edge unweighted
'''
print("siwo + same-edge unweighted")
# create integrated graph
graphs_siwo_se = []
for (p_s, p_a) in zip(c_siwo, c_virtual_same_edge):
    c_attr = p_a.values()
    c_struct = p_s.values()

    graph_integrated = utils.naive_combine(c_attr, c_struct, 0.5)
    graphs_siwo_se.append(graph_integrated)

# use Louvain to cluster
for (g, part) in zip(graphs_siwo_se, partitions_gt):
    part_louvain = community.best_partition(g)
    print("nmi, ari, ratio")
    print(normalized_mutual_info_score(part_louvain.values(), part.values()))
    print(adjusted_rand_score(part_louvain.values(), part.values()))
    print(len(set(part_louvain.values())) / len(set(part.values())))


'''
siwo + median weighted
'''
print("siwo + median weighted")
# create integrated graph
graphs_siwo_median = []
for (p_s, p_a) in zip(c_siwo, c_virtual_median):
    c_attr = p_a.values()
    c_struct = p_s.values()
    graph_integrated = utils.naive_combine(c_attr, c_struct, 0.5)
    graphs_siwo_median.append(graph_integrated)

# use Louvain to cluster
for (g, part) in zip(graphs_siwo_median, partitions_gt):
    part_louvain = community.best_partition(g)
    print("nmi, ari, ratio")
    print(normalized_mutual_info_score(part_louvain.values(), part.values()))
    print(adjusted_rand_score(part_louvain.values(), part.values()))
    print(len(set(part_louvain.values())) / len(set(part.values())))

