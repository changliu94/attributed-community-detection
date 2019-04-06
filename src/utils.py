"""
This file contains functions of utilities
"""
from __future__ import division
from sklearn.metrics.pairwise import euclidean_distances
from numpy.random import multivariate_normal as mvn
from numpy.random import seed
import numpy as np
import networkx as nx
import csv


def read_dancer_data(graph_path):
    """
    Read .graph file returned by DANCer and create a networkx graph from it.
    :param graph_path: String: the path to .graph file.
    :return: graph: Networkx Graph: the graph object created.
           : gt_community: Dictionary: the ground truth community.
    """
    gt_community = {}  # empty dictionary to store original community partition
    graph = nx.Graph()

    with open(graph_path, 'r') as f:
        fList = f.readlines()
    indices = [i for i, x in enumerate(fList) if x == "#\n"]

    vertices = fList[indices[0] + 2:indices[1]]
    edges = fList[indices[1] + 2:]

    for vertex in vertices:
        vertex = vertex.strip().split(";")
        attr = np.array([float(vertex[1]), float(vertex[2])])
        graph.add_node(int(vertex[0]), {'attr': attr})
        gt_community[int(vertex[0])] = int(vertex[3])

    for edge in edges:
        edge = edge.strip().split(";")
        graph.add_edge(int(edge[0]), int(edge[1]))

    return graph, gt_community


def redistribute_attribute(graph, gt_community, std, random_state=None):
    """
    Redistribute the attribute of the input graph.
    :param graph: Networkx Graph: the initial graph read from DANCer.
    :param gt_community: Dictionary: the ground truth community.
    :param std: Float: the standard deviation of attributes for each community.
    :param random_state: Integer: an integer value specifying the random state for mvn.
    :return: new_graph: Networkx Graph: a new object of graph with attributes.
    """
    num_community = len(set(gt_community.values()))  # number of communities
    if random_state:
        seed(random_state)
    centers = mvn([0, 0], 5*np.identity(2), num_community)
    for node in graph.nodes():
        community_id = gt_community[node]
        attr = mvn(centers[community_id], std * np.identity(2))
        graph.node[node]['attr'] = attr
    new_graph = graph.copy()

    return new_graph


def create_virtual_graphs(graph):
    """
    Create a virtual graph based on node attribute similarities.
    :param graph: Networkx Graph: the graph with attributes.
    :return: g_a: Networkx Graph: the virtual graph created based on node similarities.
    """
    n = len(graph.nodes())
    attributes = nx.get_node_attributes(graph, 'attr').values()
    D = euclidean_distances(attributes, attributes)
    D = np.exp(-D)
    edge_percentage = np.ceil(100*len(graph.edges()))/(n*(n-1)/2)
    p = np.percentile(D, 100-edge_percentage)
    D[D >= p] = 1
    D[D < p] = 0
    np.fill_diagonal(D, 0)

    graph_a = nx.from_numpy_matrix(D)
    return graph_a


def get_attr_mat(graph):
    """
    Get the attribute matrix for a given graph
    :param graph: Networkx graph.
    :return attr_mat: Numpy matrix of node attribute.
    """
    first_node = graph.nodes()[0]
    r = len(graph.node[first_node]['attr'])  # number of attributes
    attr_mat = np.zeros((len(graph), r))
    for i, node in zip(range(len(graph)), graph.nodes()):
        attr_mat[i] = graph.node[node]['attr']
    return attr_mat


def naive_combine(c_attr, c_struct, alpha):
    """
    Combine two partitions directly according to their community consensus
    :param c_attr: list: community assignment for nodes(ie, list indices) based on attribute
    :param c_struct: list: community assignment for nodes(ie, list indices) based on structure
    :param alpha: weighting parameter between structure and attribute
    :return: networkx graph: an integrated graph combing two source of information
    """
    N = len(c_attr)
    assert(len(c_attr) == len(c_struct))
    k_attr = len(set(c_attr))
    k_struct = len(set(c_struct))
    node_comm_attr = np.zeros((N, k_attr))
    node_comm_struct = np.zeros((N, k_struct))
    for i in range(N):
        node_comm_attr[i, c_attr[i]] = 1
        node_comm_struct[i, c_struct[i]] = 1
    D_struct = np.dot(node_comm_struct, node_comm_struct.T)
    D_attr = np.dot(node_comm_attr, node_comm_attr.T)
    D = alpha * D_struct + (1 - alpha) * D_attr
    D_integrated = D > 0.5
    np.fill_diagonal(D_integrated, 0)
    g_integrated = nx.from_numpy_matrix(D_integrated)
    return g_integrated


def compute_within_inertia(graph, part):
    """
    Compute the witin ineratia of a graph and its partition
    :param graph: networkx graph object
    :param part: dictionary: the community partition over graph
    :return: float: the within inertia ratio
    """
    X = get_attr_mat(graph)
    g_g = np.mean(X, axis=0)  # gravity center of the whole graph
    g_g = g_g.reshape(1, -1)
    denom = np.sum(np.power(np.linalg.norm(X - g_g, axis=1), 2))  # denominator of the graph
    nb_comm = len(set(part.values()))  # number of communities
    num = 0.0  # initialize the numerator to 0
    for i in range(nb_comm):
        X_c = X[np.equal(part.values(), i), :]  # subset the attribute matrix with nodes in community i
        g_c = np.mean(X_c, axis=0)  # the gravity center of community i
        g_c = g_c.reshape(1, -1)
        num += (X_c.shape[0] / X.shape[0]) * np.sum(np.power(np.linalg.norm(X_c - g_c, axis=1), 2))
    return float(num / denom)


def read_fb_community(path):
    """
    Read ground truth community of facebook dataset
    :param path: String: the path to the ground truth community
    :return: Dict: the ground truth community
    """
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)
    gt_community = {}
    for i in range(len(d)):
        for j in range(1, len(d[i])):
            gt_community[int(d[i][j])] = i
    return gt_community


def read_fb_edges(path):
    """
    Read edges of facebook/sinanet dataset
    :param path: String: the path to the edge data
    :return: List: a list of 2-element lists containing undirected edges
    """
    edge_list = []
    with open(path, 'r') as f:
        for line in f:
            line = [int(i) for i in line.split()]
            edge_list.append(line)
    return edge_list


def read_fb_data(path_prefix):
    """
    Read and process facebook data, create Networkx graph objects.
    :param path_prefix: String: the prefix of the path to the network dataset.
    :return: graph: Networkx Graph: the graph object created.
           : gt_community: Dictionary: the ground truth community.
    """
    edge_path = path_prefix + '.edges'
    gt_community_path = path_prefix + '.circles'
    feat_path = path_prefix + '.feat'

    graph = nx.Graph()
    with open(feat_path, 'r') as f:
        for line in f:
            line = [float(i) for i in line.split()]
            graph.add_node(int(line[0]), {'attr': np.array(line[1:])})

    edge_list = read_fb_edges(edge_path)
    for edge in edge_list:
        graph.add_edge(edge[0], edge[1])

    gt_community = read_fb_community(gt_community_path)

    return graph, gt_community


def add_outliers(graph, gt_community):
    """
    Add outliers into the ground truth patition dictionary and assign value -1
    :param graph: networkx graph: a graph with outlier nodes
    :param gt_community: Dictionary: ground truth partition
    :return: gt_community: Dictionary: ground truth partition added outliers
    """
    for node in graph.nodes():
        if node not in gt_community:
            gt_community[node] = -1

    return gt_community


def read_sina_community(path):
    """
    Read ground truth community of sinanet dataset
    :param path: String: the path to the ground truth community
    :return: Dict: the ground truth community
    """
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)
    gt_community = {}
    for i in range(len(d)):
        for j in range(len(d[i])):
            if int(d[i][j]) == 0:
                break
            else:
                gt_community[int(d[i][j])] = i
    return gt_community


def read_sina_data(path_prefix):
    """
    Read and process sinanet data, create Networkx graph objects.
    :param path_prefix: String: the prefix of the path to the network dataset.
    :return: graph: Networkx Graph: the graph object created.
           : gt_community: Dictionary: the ground truth community.
    """
    edge_path = path_prefix + 'edge.txt'
    gt_community_path = path_prefix + 'clusters.txt'
    feat_path = path_prefix + 'content.txt'

    graph = nx.Graph()
    with open(feat_path, 'r') as f:
        node = 1
        for line in f:
            line = [float(i) for i in line.split()]
            graph.add_node(node, {'attr': np.array(line)})
            node += 1

    edge_list = read_fb_edges(edge_path)
    for edge in edge_list:
        graph.add_edge(edge[0], edge[1])

    gt_community = read_sina_community(gt_community_path)

    return graph, gt_community


def fit_IL(graph):
    """
    Fit Ilouvain
    :param graph: input networkx graph object
    :return: attributes and author_index for IL algorithm
    """
    attributes = {}
    author_index = {}
    for n in graph:
        author_index[n] = n
        attr = graph.node[n]['attr']
        attr_IL = {}
        for i, attr_value in enumerate(attr):
            attr_IL[i] = attr_value
        attributes[n] = attr_IL

    return attributes, author_index
