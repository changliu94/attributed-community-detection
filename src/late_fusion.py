"""This script implements the late fusion algorithm
"""
from __future__ import division
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from src import SIWO, community


class LateFusionNumeric:
    def __init__(self, graph_orig, attr_mat, alg_struct, alg_attr, alpha):
        self.graph = graph_orig
        self.num_nodes = len(graph_orig.nodes())
        self.A = attr_mat
        self.alg_struct = alg_struct
        self.alg_attr = alg_attr
        self.alpha = alpha

    def get_communities(self, target_graph=None):
        """
        Find the structural communities of the original graph
        :param target_graph: the target graph to find communities
        :return: the structural communities
        """
        if target_graph is None:
            target_graph = self.graph.copy()

        if self.alg_struct == "louvain":
            part_struct = community.best_partition(target_graph)
        elif self.alg_struct == "siwo":
            part_struct = SIWO.best_partition(target_graph)
        else:
            raise NameError("Algorithm is not implemented")
        part_struct = part_struct.values()
        assert(isinstance(part_struct, list))
        return part_struct

    def get_clusters(self, num_clusters=None):
        """
        Find the clusters based on node attributes
        :return: clusters found by the clustering algorithm
        """
        if self.alg_attr == "kmeans":
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.A)
            clusters = kmeans.labels_
        elif self.alg_attr == "dbscan":
            dist = euclidean_distances(self.A, self.A)
            avg_degree = np.mean(self.graph.degree().values())
            median_dist = np.median(dist)
            db = DBSCAN(eps=median_dist / 20, min_samples=avg_degree)
            clusters = db.fit_predict(self.A)
        elif self.alg_attr == "sc":
            dist = euclidean_distances(self.A, self.A)
            sim = np.exp(-dist)
            sc = SpectralClustering(num_clusters, affinity='precomputed', n_init=20, assign_labels='discretize')
            clusters = sc.fit_predict(sim)
        else:
            raise NameError("Algorithm is not implemented")
        return clusters

    def get_affiliation_matrix(self, part):
        """
        Build the node-community affiliation matrix from a partitioning
        :param part: list: the list of community assignment for each node,
        the values must be integers starting from 0
        :return: the node-community affiliation matrix
        """
        num_attr = len(set(part))
        affiliation_mat = np.zeros((self.num_nodes, num_attr))
        for i in range(self.num_nodes):
            affiliation_mat[i, part[i]] = 1
        return affiliation_mat

    def late_fusion(self, num_clusters):
        """
        Combine two partitionings to produce the final partitioning
        :param num_clusters: the number of clusters that should be found by the clustering algorithm
        :return the final partitioning
        """
        part_struct = self.get_communities()
        part_attr = self.get_clusters(num_clusters)
        affiliation_struct = self.get_affiliation_matrix(part_struct)
        affiliation_attr = self.get_affiliation_matrix(part_attr)
        adjacency_struct = np.dot(affiliation_struct, affiliation_struct.T)
        adjacency_attr = np.dot(affiliation_attr, affiliation_attr.T)
        adjacency = self.alpha * adjacency_struct + (1-self.alpha) * adjacency_attr
        np.fill_diagonal(adjacency, 0)
        graph_int = nx.from_numpy_matrix(adjacency)
        part_final = self.get_communities(graph_int)
        return part_final


class LateFusionBinary:
    def __init__(self, graph_orig, attr_mat, alg_struct, alpha):
        self.graph = graph_orig
        self.num_nodes = len(graph_orig.nodes())
        self.A = attr_mat
        self.alg_struct = alg_struct
        self.alpha = alpha

    def get_communities(self, target_graph=None):
        """
        Find the structural communities of the original graph
        :param target_graph: the target graph to find communities
        :return: the structural communities
        """
        if target_graph is None:
            target_graph = self.graph.copy()

        if self.alg_struct == "louvain":
            part_struct = community.best_partition(target_graph)
        elif self.alg_struct == "siwo":
            part_struct = SIWO.best_partition(target_graph)
        else:
            raise NameError("Algorithm is not implemented")
        part_struct = part_struct.values()
        assert(isinstance(part_struct, list))
        return part_struct

    def create_virtual_graph(self, method, weighted):
        """ Create virtual graph based on attribute similarity
        :param method: str specifying the thresholding method, equal-edge or median
        :param weighted: boolean indicating whether the virtual graph should be weighted or not
        :return: the virtual graph
        """
        sim = np.dot(self.A, self.A.T)
        triu_inds = np.triu_indices(len(sim), 1)
        sim_vec = sim[triu_inds]
        if method == 'equal':
            num_edges = len(self.graph.edges())
            total_possible_edges = self.num_nodes * (self.num_nodes-1) / 2
            quantile = 1 - num_edges / total_possible_edges
            threshold = np.quantile(sim_vec, quantile)
        elif method == 'median':
            threshold = np.median(sim_vec)
        else:
            raise NameError('Not a valid thresholding method')

        sim[sim <= threshold] = 0
        if weighted:
            sim[sim > threshold] = 1
        np.fill_diagonal(sim, 0)
        graph_attr = nx.from_numpy_matrix(sim)

        return graph_attr

    def get_affiliation_matrix(self, part):
        """
        Build the node-community affiliation matrix from a partitioning
        :param part: list: the list of community assignment for each node,
        the values must be integers starting from 0
        :return: the node-community affiliation matrix
        """
        num_attr = len(set(part))
        affiliation_mat = np.zeros((self.num_nodes, num_attr))
        for i in range(self.num_nodes):
            affiliation_mat[i, part[i]] = 1
        return affiliation_mat

    def late_fusion(self, method, weighted):
        """
        Combine two partitionings to produce the final partitioning
        :return: the final partitioning
        """
        part_struct = self.get_communities()
        graph_attr = self.create_virtual_graph(method, weighted)
        part_attr = self.get_communities(target_graph=graph_attr)
        affiliation_struct = self.get_affiliation_matrix(part_struct)
        affiliation_attr = self.get_affiliation_matrix(part_attr)
        adjacency_struct = np.dot(affiliation_struct, affiliation_struct.T)
        adjacency_attr = np.dot(affiliation_attr, affiliation_attr.T)
        adjacency = self.alpha * adjacency_struct + (1-self.alpha) * adjacency_attr
        np.fill_diagonal(adjacency, 0)
        graph_int = nx.from_numpy_matrix(adjacency)
        part_final = self.get_communities(graph_int)
        return part_final
