# -*- coding: utf-8 -*-
'''
    Copyright (C) 2009 by
    Thomas Aynaud <thomas.aynaud@lip6.fr>
    All rights reserved.
    BSD license: http://python-louvain.readthedocs.io/en/latest/
    
'''

'''
This module implements SIWO community detection as published in the paper "???".
This code is derived from python-louvain available at: https://github.com/taynaud/python-louvain
'''
import networkx as nx
import array
import argparse
import numpy as np
import os.path
from math import log
from community_status import Status
import time
from random import shuffle



__PASS_MAX = -1
__MIN = 0.0000001



def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities

    Parameters
    ----------
    dendrogram : list of dict
       a list of partitions, ie dictionnaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]

    Returns
    -------
    partition : dictionnary
       A dictionary where keys are the nodes and the values are the set it
       belongs to

    Raises
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high

    See Also
    --------
    best_partition which directly combines partition_at_level and
    generate_dendrogram to obtain the partition of highest modularity

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition

def best_partition(graph, partition=None, weight='weight', weighted = False):
    """
    find communities in graph based on SIWO algorithm 

    Parameters
    ----------
    graph : networkx.Graph
    partition : dict, optional
       the algorithm will start using this partition of the nodes.
       It's a dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'

    Returns
    -------
    partition : dictionnary
       The partition, with communities numbered from 0 to number of communities

    Raises
    ------
    NetworkXError
       If the graph is not Eulerian.

    See Also
    --------
    generate_dendrogram to obtain all the decompositions levels

    Notes
    -----
    Uses Louvain's optimization algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in
    large networks. J. Stat. Mech 10008, 1-12(2008).
    2. reference to SIWO algorithm TODO

    Examples
    --------
    >>>  #Basic usage
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> part = best_partition(G)

    >>> #other example to display a graph with its community :
    >>> #better with karate_graph() as defined in networkx examples
    >>> #erdos renyi don't have true community structure
    >>> G = nx.erdos_renyi_graph(30, 0.05)
    >>> #first compute the best partition
    >>> partition = community.best_partition(G)
    >>>  #drawing
    >>> size = float(len(set(partition.values())))
    >>> pos = nx.spring_layout(G)
    >>> count = 0.
    >>> for com in set(partition.values()) :
    >>>     count += 1.
    >>>     list_nodes = [nodes for nodes in partition.keys()
    >>>                                 if partition[nodes] == com]
    >>>     nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))
    >>> nx.draw_networkx_edges(G,pos, alpha=0.5)
    >>> plt.show()
    """
    
    
    # check networkx version
    if nx.__version__!='1.11': # Base Graph Class has changed With the release of NetworkX 2.0 !!!
        print("This program requires networkx 1.11")
        exit(-1)
    
    #---------------------Step1. Pre-Processing------------------------------
    graph = graph.copy()
    Dangles = PreProcessing(graph, weighted, weight)
    
    #---------------------Step2. Optimizing SIWO-----------------------------
    
    dendo = generate_dendrogram(graph, __IncSIWO, partition, weight)
    partition = partition_at_level(dendo, len(dendo) - 1)
    
    
    #-----------------Step3. Core/non core Community aggregation-------------
    
    graph_cpy = graph.copy()
    RemoveSingles(graph, partition) # temporarily remove Lone communities
   
    RemoveWeights(graph)
    Ngraph = induced_graph(partition, graph, weight)
    dendo = generate_dendrogram(Ngraph, __IncOutDegree, None, weight )
    
    dendoN=[partition]
    dendoN.extend(dendo)
    partition=partition_at_level(dendoN, len(dendoN) - 1)
    
    
    #---------------------Step4. Post-Processing-----------------------------
    
    graph = graph_cpy
    AddSingles(partition,graph)
    partition = __renumber(partition)
    
    AddDangles(graph, partition, Dangles)
    
    return partition


def generate_dendrogram(graph, IncFnc, part_init=None, weight='weight'):
    """Find communities in the graph and return the associated dendrogram

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger
    are the communities


    Parameters
    ----------
    graph : networkx.Graph
        the networkx graph which will be decomposed
    part_init : dict, optional
        the algorithm will start using this partition of the nodes. It's a
        dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    dendrogram : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph

    Raises
    ------
    TypeError
        If the graph is not a networkx.Graph

    See Also
    --------
    best_partition

    Notes
    -----
    Uses Louvain's optimization algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in large
    networks. J. Stat. Mech 10008, 1-12(2008).
    .. 2. reference to SIWO paper TODO
    
    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendo = generate_dendrogram(G)
    >>> for level in range(len(dendo) - 1) :
    >>>     print("partition at level", level,
    >>>           "is", partition_at_level(dendo, level))
    :param weight:
    :type weight:
    """
    
    if type(graph) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for node in graph.nodes():
            part[node] = node
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    
    changed = __one_level(current_graph, IncFnc, status, weight)
    partition = __renumber(status.node2com)
        
    status_list.append(partition)
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        changed = __one_level(current_graph, IncFnc, status, weight)
        if not changed:
            break    
        partition = __renumber(status.node2com)
        status_list.append(partition)
        
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
    
    return status_list[:]


def induced_graph(partition, graph, weight="weight"):
    """Produce the graph where nodes are the communities

    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w

    Parameters
    ----------
    partition : dict
       a dictionary where keys are graph nodes and  values the part the node
       belongs to
    graph : networkx.Graph
        the initial graph
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    g : networkx.Graph
       a networkx graph where nodes are the parts

    Examples
    --------
    >>> n = 5
    >>> g = nx.complete_graph(2*n)
    >>> part = dict([])
    >>> for node in g.nodes() :
    >>>     part[node] = node % 2
    >>> ind = induced_graph(part, g)
    >>> goal = nx.Graph()
    >>> goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])  # NOQA
    >>> nx.is_isomorphic(int, goal)
    True
    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges_iter(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, attr_dict={weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    count = 0
    ret = dictionary.copy()
    new_values = dict([])

    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:
            new_values[value] = count
            new_value = count
            count += 1
        ret[key] = new_value

    return ret


def RemoveWeights(graph, weight="weight"):
    for u,v in graph.edges():
        graph.add_edge(u, v, attr_dict={weight: 1})    
    
def shared_neighbors_cnt(graphBase,u,v):
    shared = 0
    if(graphBase.degree(u) > graphBase.degree(v)):
        tmp = u
        u = v
        v = tmp
    neighbors_u =  graphBase[u]
    neighbors_v =  graphBase[v]
    for n1 in neighbors_u:
        if n1 in neighbors_v:
            shared = shared+1
    return shared    
        
def get_mutuals(graphBase):
    mutuals = {}
    maxMutuals = {}
    totalMutuals = {}
    nodes = graphBase.nodes()
    nbVertices = len(nodes)
    
    for i in range(nbVertices):
        mutuals[nodes[i]] = {}
        maxMutuals[nodes[i]] = -1
        totalMutuals[nodes[i]] = 0.0
    
    for i in range(nbVertices):
        neighbors = graphBase.neighbors(nodes[i])
        for neigh in neighbors:
            if neigh in mutuals[nodes[i]]:
                continue
            curMutual = shared_neighbors_cnt(graphBase,nodes[i],neigh)
            mutuals[nodes[i]][neigh] = curMutual
            mutuals[neigh][nodes[i]] = curMutual
            
            totalMutuals[nodes[i]] = totalMutuals[nodes[i]] + curMutual
            totalMutuals[neigh] = totalMutuals[neigh] + curMutual
            
            if curMutual > maxMutuals[nodes[i]]:
                maxMutuals[nodes[i]] = curMutual
            if curMutual > maxMutuals[neigh]:
                maxMutuals[neigh] = curMutual    
            
        totalMutuals[nodes[i]] = totalMutuals[nodes[i]]/2.0    
        
    return mutuals, maxMutuals, totalMutuals
    
def Clustering_Coef(graphBase, totalMutuals):
    Clust_Coef = {}
    nodes = graphBase.nodes()
    for node in nodes:        
        totalMutual = totalMutuals[node]    
        deg = graphBase.degree(node)
        possibleTri = (deg * (deg-1)) / 2
        if possibleTri==0:
            Clust_Coef[node] = 1
        else:
            Clust_Coef[node] = totalMutual/ float(possibleTri)
    return Clust_Coef    

def ComputeEdgeStrength(graphBase, weight="weight"):
    nodes = graphBase.nodes()
    nbVertices = len(nodes)
    mutuals, maxMutuals ,totalMutuals = get_mutuals(graphBase)
    CCoef = Clustering_Coef(graphBase, totalMutuals)
    for i in range(nbVertices):
        neighbors = graphBase.neighbors(nodes[i])
        maxMutual = maxMutuals.get(nodes[i])
            
        for neigh in neighbors:
            if (CCoef.get(nodes[i]) < CCoef.get(neigh)):
                continue
                
            CurMutuals = mutuals.get(nodes[i])
            #length of each bin
            binlen = 2/(float)(maxMutual+1)
            # min point of the bin 
            minpoint = -1 + (CurMutuals.get(neigh) * binlen )
            # max point of the bin 
            maxpoint = minpoint + binlen
            #average point of the bin
            avgpoint = (minpoint + maxpoint)/ 2.0
            w = avgpoint
            
            graphBase.add_edge(nodes[i], neigh, attr_dict={weight: w})

    return CCoef
def NormalizeWeights(graph, CCoef, weight="weight"):
    nodes = graph.nodes()
    nbVertices = len(nodes)
    for i in range(nbVertices):
        node = nodes[i]
        maxWeight = -float('Inf')
        minWeight = float('Inf')
        for neighbor, datas in graph[node].items(): # compute the max weight for each node
            edge_weight = datas.get(weight, 1)
            if edge_weight>maxWeight:
                maxWeight = edge_weight
            if edge_weight<minWeight:
                minWeight = edge_weight    
        for neighbor, datas in graph[node].items():
            if (CCoef.get(nodes[i]) < CCoef.get(neighbor)):
                continue
            # scale edge weight to [0,1]
            edge_weight = datas.get(weight, 1)
            if maxWeight==minWeight:
                w = 1.0/len(graph.neighbors(node))
            else:    
                w = float(edge_weight - minWeight)/(maxWeight-minWeight)  
              
            graph.add_edge(nodes[i], neighbor, attr_dict={"w_norm": w})

def CombineWeights(graph, weight="weight"):
    nodes = graph.nodes()
    nbVertices = len(nodes)
    for i in range(nbVertices):
        node = nodes[i]
        for neighbor, datas in graph[node].items(): # combine weight and strength of each edge
            if neighbor < node:
                continue
            edge_weight = datas.get("w_norm", 1)
            edge_strength = datas.get("strength")
            w = (edge_strength+1)* edge_weight-1
            graph.add_edge(nodes[i], neighbor, attr_dict={weight: w})
                
def getSingleNodes(partition):
    com2node = {}
    singleNodes = []
    nodes = partition.keys()
    coms = set(partition.values())
    for com in coms:
        com2node[com] = []
        
    for node in nodes:
        com = partition[node]
        com2node[com].append(node)
    for com in coms:
        if len(com2node[com])==1:
            singleNodes.append(com2node[com][0])
    return singleNodes,com2node    

def degree_centrality(node, comSize, partition, graph):
    neighbors = graph.neighbors(node)
    connected = 0.0
    for neigh in neighbors:
        com_neigh = partition[neigh]
        
        if com_neigh == partition[node]:
            connected += 1
    return connected/ float(comSize-1)        
    

def AddSingles(partition, graphBase):
    SingleNodes,com2node = getSingleNodes(partition)
    SingleNodes_cpy = list(SingleNodes)

    modified = True
    ToBeRmvd = []
    
    if len(SingleNodes)==0:
        return partition
        
           
    while(modified):
        modified = False
        
        for node in ToBeRmvd:
            SingleNodes.remove(node)
            
        ToBeRmvd = []    
        for node in SingleNodes_cpy:
            com_node = partition[node]
            bestCom = com_node
            maxComSize = 0
            neighbors = graphBase.neighbors(node)
            neighComSize = {} #map between neighbor community and its community size
            
            for neigh in neighbors:
                
                if neigh in SingleNodes:
                    continue
                    
                neighcom = partition[neigh]
                if neighcom not in neighComSize :
                    neighComSize[neighcom] = 1
                else:
                    neighComSize[neighcom] += 1    
                    
            if len(neighComSize)==0:
                continue
            maxComSize = max(neighComSize.values())
            minComSize = min(neighComSize.values())    
            if(maxComSize==1):
                # choose neighbor node with largest degree centrality
                maxDC=0
                for neigh in neighbors:
                    if neigh in SingleNodes:
                        continue
                    neighcom = partition[neigh]    
                    DC = degree_centrality(neigh, len(com2node[neighcom]), partition, graphBase)
                    if DC >= maxDC:
                        maxDC = DC
                        bestCom = partition[neigh]        
            else:            
                for com, size in neighComSize.iteritems():
                    if size == maxComSize:
                        bestCom = com
                        break
                        
            partition[node] = bestCom
            if(bestCom != com_node):
                if(node in SingleNodes):
                    ToBeRmvd.append(node)    
                modified = True
    if len(SingleNodes)!=0: # we were not able to add some of the single nodes to the graph (for example when the graph is a circle of nodes)
        com = max(com2node.keys())
        for node in SingleNodes:
            partition[node] = com # put each in its own community
            com += 1
            
                     
    return partition


def RemoveDangles(graph, candidates):
    Dangles = {}
    for node in candidates:
        deg = graph.degree(node)
        if deg==1:
            neighbor = graph.neighbors(node)[0]
            Dangles[node] = neighbor
        else:
            continue       
    Tobermvd = []
    for node in Dangles:
        if graph.degree(node)==0:
            Tobermvd.append(node)
            continue
        graph.remove_node(node)
    for node in Tobermvd:
        del Dangles[node]
    return Dangles            

def RemoveSingles(graph, partition):
    singles, com2node = getSingleNodes(partition) 
    for node in singles:
        graph.remove_node(node)

def PreProcessing(graph, weighted, weight):
    
    if weighted:
        CCoef = ComputeEdgeStrength(graph, weight='strength')
        NormalizeWeights(graph, CCoef)
        CombineWeights(graph)
    else:
        ComputeEdgeStrength(graph, weight)
    AllDangles = []
    Dangles = RemoveDangles(graph, graph.nodes())
    while(Dangles):
        AllDangles.append(Dangles)
        Dangles = RemoveDangles(graph, Dangles.values())
    return AllDangles
    
def AddDangles(graph, partition, AllDangles):
    i = len(AllDangles)-1
    while i >= 0:
        Dangles = AllDangles[i]
        for u,v in Dangles.iteritems():
            partition[u] = partition[v]
        i = i-1


def __IncSIWO(graph, status, neigh_communities, com_neigh, com_node, node):
    return neigh_communities.get(com_neigh, 0.) - neigh_communities.get(com_node, 0.)

def __IncOutDegree(graph, status, neigh_communities, com_neigh, com_node, node):
    
    # in_degrees and out_degrees before and after moving node from com_node to com_neigh
    
    inDeg_node_b = status.loops.get(node, 0.) * 2.0
    outDeg_node_b = status.gdegrees[node] - inDeg_node_b
    
    #inDeg_node_a = inDeg_node_b + neigh_communities.get(com_neigh, 0.)*2
    outDeg_node_a = outDeg_node_b - neigh_communities.get(com_neigh, 0.)
    
    '''
    # Def 1
    if((inDeg_node_b) > outDeg_node_b):
        return 0
        
    incr =     outDeg_node_b - outDeg_node_a
    
    
    #print ("after "+ str(LogAfter))
    #print("before "+ str(LogBefore))
    return incr
    '''
    '''
    # Def 2
    if((inDeg_node_b/2.0) > outDeg_node_b):
        return 0
        
    incr =     outDeg_node_b - outDeg_node_a
    
    
    #print ("after "+ str(LogAfter))
    #print("before "+ str(LogBefore))
    return incr
    '''
    
    '''
    # Def 3 
    incr = neigh_communities.get(com_neigh, 0.) - (inDeg_node_b / 2.0)
    return max(incr, 0)
    '''
    
    
    # Def 4: 1 & 3 
    between_links = neigh_communities.values()
    within_links = inDeg_node_b / 2.0
    max_between_links = max(between_links)
    if  inDeg_node_b > outDeg_node_b and within_links > max_between_links:
        return 0
    incr =     outDeg_node_b - outDeg_node_a
    return     incr
    
    

def __one_level(graph, IncFnc, status, weight_key):
    """Compute one level of communities
    """
    modified = True
    changed = False
    nb_pass_done = 0
    graph_nodes = graph.nodes()
    shuffle(graph_nodes)
    while modified and nb_pass_done != __PASS_MAX:
        modified = False
        nb_pass_done += 1
        for node in graph_nodes:
            com_node = status.node2com[node]
            neigh_communities = __neighcom(node, graph, status, weight_key)
            
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in neigh_communities.items():
                incr = IncFnc(graph, status, neigh_communities, com, com_node, node)
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
             
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True
                changed = True    
        return changed    


def __neighcom(node, graph, status, weight_key):
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights
    
def __neighcom(node, graph, status, weight_key) :
    
    weights = {}
    voisins = graph[node].iteritems()
    curCommunity = status.node2com[node]

    for neighbor, datas in voisins :
        if neighbor != node :
            weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom,0)+ weight

    return weights    


def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))

def __SIWO(status):
    """
    Fast compute the value of SIWO measure for the partition of the graph using
    status precomputed
    """
    total = 0.
    for community in set(status.node2com.values()):
        total += status.internals.get(community, 0.)
    return total
    
def loadDataset(path, weighted):
    graph = nx.Graph()

    # Read the graph
    if(not os.path.isfile(path)):
        print("Error: file " + path + "not found")
        exit(-1)
    with open(path ) as f:
        for line in f.readlines():
            if weighted:
                w =float(line.split(";")[1])
                v1 = int((line.split(";")[0]).split(" ")[0])
                v2 = int((line.split(";")[0]).split(" ")[1])
                graph.add_node(v1)
                graph.add_node(v2)
                graph.add_edge(v1, v2, attr_dict={"weight":w})
            else:
                if ";" in line:
                    line = line.split(";")[0]    
                v1 = int(line.split(" ")[0])
                v2 = int(line.split(" ")[1])
                graph.add_node(v1)
                graph.add_node(v2)
                graph.add_edge(v1, v2)

    # Build authorIndex
    authorIndex = {}
    for n in graph:
        authorIndex[n] = n
    
    return graph, authorIndex

def __main() :
    
    starttot = time.time()
    parser = argparse.ArgumentParser(description = 'The SIWO algorithm', prog = "SIWO")
    parser.add_argument('dataset', help='Path to the network files')
    parser.add_argument('output', help='output name')
    parser.add_argument('-w', help='weighted graphs',action="store_true")
    args = parser.parse_args()
    graph, authorIndex = loadDataset(args.dataset, args.w)
    #print(len(graph.nodes()))
    partition = best_partition(graph, weighted=args.w)
    # write output
    #print(len(partition))
    f = open("results/"+args.output ,'w')
    for elem, part in sorted(partition.iteritems()):
        out = str(elem) + " " + str(part)
        f.write(out + "\n")
    f.close()
    endtot = time.time()
    #print("Total time "+str(endtot-starttot))

if __name__ == "__main__" :
    __main()

