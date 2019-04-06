#!/usr/bin/python

from __future__ import division

from pprint import pprint

import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os.path
import time

# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
__all__ = ["partition_at_level", "modularity", "best_partition", "generate_dendogram", "induced_graph"]
__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.

import networkx as nx

class ML2:
    __MIN = 0.000001
    __PASS_MAX = -1
    LOGOPERATIONS = False

    nbVertices = 0

    def __init__(self, graph, attributes, authorIndex):
        self.graph = graph
        self.graphBase = graph.copy()
        self.attributes = attributes
        self.nbVertices = len(graph)
        self.statusTab = []
        self.authorIndex = authorIndex

        # Build status structures
        status = Status()
        status.init(graph)
        self.statusTab.append(status)
        statusA = Status()
        statusA.initAttribStatus(graph, authorIndex, attributes)
        self.statusTab.append(statusA)

        self.status_list = list()

    def critereCombinaison(self):
        # if(args.verbose):
        #     print "Mod1: " + str(self.__modularity(self.statusTab[0]))
        #     print "Mod2: " + str(self.__modularity(self.statusTab[1]))
        return (self.__modularity(self.statusTab[0]) + self.__modularity(self.statusTab[1])) / 2.

    def findPartition(self) :
        giniMatrix = self.calculateGiniMatrixInitial()
        self.__one_level(giniMatrix=None)
        new_mod = self.critereCombinaison()

        partition, bijection = self.__renumber()

        self.status_list.append(partition)
        mod = new_mod
        self.induced_graph(partition)
        self.statusTab[0].init(self.graph)

        # if(args.verbose):
        #     print "Inducing attrib status"
        self.statusTab[1].inducedAttribStatusTab(partition, bijection)
        giniMatrix = self.firstInducedGiniMatrix(partition, giniMatrix)

        while True :
            self.__one_level(giniMatrix=giniMatrix)
            new_mod = self.critereCombinaison()
            if new_mod - mod < self.__MIN :
                # if(args.verbose):
                #     print "modularities"
                #     print self.__modularity(self.statusTab[0])
                #     print self.__modularity(self.statusTab[1])
                #     print "Modularity Final: " + str(self.__modularity(self.statusTab[1]) + self.__modularity(self.statusTab[0]))
                break
            partition, bijection  = self.__renumber()

            self.status_list.append(partition)

            mod = new_mod
            self.induced_graph(partition)
            giniMatrix = self.inducedGiniMatrix(partition, giniMatrix)

            self.statusTab[0].init(self.graph)

            if self.statusTab[1] != None :
                self.statusTab[1].inducedAttribStatusTab(partition, bijection)

        dendogram = self.status_list[:]

        # Generate and output partition
        partition = dendogram[0].copy()
        for index in range(1, len(dendogram)) :
            for node, community in partition.iteritems() :
                partition[node] = dendogram[index][community]
        # for elem, part in sorted(partition.iteritems()) :
        #     if(args.verbose):
        #         print str(self.authorIndex[elem]) + " " + str(part) + " " + str(self.attributes[self.authorIndex[elem]])
        #     else:
        #         out = str(self.authorIndex[elem]) + " " + str(part)
        #         if(args.multipleDataset != None):
        #             f = open(args.dataset + "_" + str(curDatasetIdx) + ".2ModLouvain",'a')
        #         else:
        #             f = open(args.dataset + ".2ModLouvain",'a')
        #         f.write(out + "\n")
        #         f.close()
        return partition

    def dist(self, v1, v2):
        attrV1 = self.attributes[v1]
        attrV2 = self.attributes[v2]
        distance = 0.
        for attr, val1 in attrV1.items():
            val2 = attrV2.get(attr, 0.)
            distance += (val1 - val2)**2
        for attr, val2 in attrV2.items():
            if not attr in attrV1:
                distance += val2 * val2
        return distance

    def distArray(self, v1, v2):
        attrV1 = self.attributes[v1]
        attrV2 = self.attributes[v2]
        distance = 0.
        for i in range(len(attrV1)):
            distance += (attrV1[i] - attrV2[i])**2
        return distance

    def firstInducedGiniMatrix(self, partition, giniMatrix):
        out = np.zeros([len(set(partition.values())), len(set(partition.values()))])
        # if(args.verbose):
        #     pprint(giniMatrix)
        for i in partition:
            for j in partition:
                out[partition[i]][partition[j]] = giniMatrix[self.authorIndex[i]][self.authorIndex[j]]
        return out

    def inducedGiniMatrix(self, partition, giniMatrix):
        # if(args.verbose):
        #     print "inducedGiniMatrix..."
        out = np.zeros([len(set(partition.values())), len(set(partition.values()))])
        for i in partition:
            for j in partition:
                out[partition[i]][partition[j]] = out[partition[i]][partition[j]] + giniMatrix[i][j]
        # if(args.verbose):
        #     print "End inducedGiniMatrix"

        return out

    def calculateGiniMatrixInitial(self):
        giniMatrix = {}
        for v1 in self.graph:
            giniMatrix[self.authorIndex[v1]] = {}
        np.zeros(self.nbVertices ** 2).reshape((self.nbVertices, self.nbVertices))
        for v1 in self.graph:
            for v2 in self.graph:
                d = -1 * self.dist(self.authorIndex[v1], self.authorIndex[v2]) / self.nbVertices**2
                giniMatrix[self.authorIndex[v1]][self.authorIndex[v2]] = d
                giniMatrix[self.authorIndex[v2]][self.authorIndex[v1]] = d
        """
        print "Calculating Gini Matrix Initial"
        Y = pdist(self.attributes, 'sqeuclidean')
        print "division"
        Y = np.divide(Y, 0.0-float(len(self.attributes)**2))
        pprint(giniMatrix)
        pprint(squareform(Y))
        return squareform(Y)
        """
        return giniMatrix


    def induced_graph(self, partition) :
        newGraph = nx.Graph()
        newGraph.add_nodes_from(partition.values())

        for node1, node2, datas in self.graph.edges_iter(data = True) :
            weight = datas.get("weight", 1)
            com1 = partition[node1]
            com2 = partition[node2]
            w_prec = newGraph.get_edge_data(com1, com2, {"weight":0}).get("weight", 1)
            newGraph.add_edge(com1, com2, weight = w_prec + weight)
        self.graph = newGraph

    def __renumber(self) :
        count = 0
        dictionary = self.statusTab[0].node2com
        ret = dictionary.copy()
        new_values = dict([])
        for key in dictionary.keys() :
            value = dictionary[key]
            new_value = new_values.get(value, -1)
            if new_value == -1 :
                new_values[value] = count
                new_value = count
                count = count + 1
            ret[key] = new_value
        return ret, new_values

    def __one_level(self, giniMatrix = None) :
        modif = True

        while modif:
            modif = False
            numNode = 0
            for node in self.graph.nodes() :
                numNode = numNode + 1

                com_node = self.statusTab[0].node2com[node]
                best_com = com_node

                best_increase = 0
                neigh_communities = self.__neighcom(node, giniMatrix=giniMatrix)
                # if(args.verbose):
                #     print "Neighb Communities of " + str(node)
                #     pprint(neigh_communities)

                degc_totw_tab = []

                for i in range(len(self.statusTab)):
                    
                    degc_totw_tab.append( self.statusTab[i].gdegrees.get(node, 0.) / (self.statusTab[i].total_weight*2.) )
                    theWeight = neigh_communities[com_node][i]

                    if abs(self.statusTab[i].degrees[com_node]) <= abs(self.statusTab[i].gdegrees[node]):
                        self.statusTab[i].degrees[com_node] = abs(self.statusTab[i].gdegrees[node])

                    self.__remove(node, com_node, theWeight, self.statusTab[i])
                assert(self.statusTab[0].node2com[node] == self.statusTab[1].node2com[node])

                # Find the best community
                for com, dnc in neigh_communities.iteritems() :
                    incr = 0.
                    for i in range(len(self.statusTab)):
                        totw = abs(self.statusTab[i].total_weight)
                        if i == 0:
                            a = ( abs(dnc[i])  - abs(self.statusTab[i].degrees.get(com, 0.) * degc_totw_tab[i] )) / totw
                            incr += a
                        else:
                            a = (0.0 - abs( dnc[i])  + abs(self.statusTab[i].degrees.get(com, 0.) * degc_totw_tab[i])) / totw
                            incr += a
                    incr /= 2
                    if incr > best_increase :
                        best_increase = incr
                        best_com = com

                for i in range(len(self.statusTab)):
                    if best_com in neigh_communities:
                        theWeight = neigh_communities[best_com][i]
                    else:
                        print "IS THAT POSSIBLE ???? (best_com not in neigh_communities)"
                        exit(0)
                        theWeight = 0
                    self.__insert(node, best_com, theWeight, self.statusTab[i])

                if best_com != com_node :
                    modif = True

    def __neighcom(self, node, giniMatrix = None) :
        weights = {}
        voisins = self.graph[node].iteritems()
        curCommunity = self.statusTab[0].node2com[node]
        if curCommunity not in weights :
            weights[curCommunity] = np.zeros([len(self.statusTab)])

        for neighbor, datas in voisins :
            if neighbor != node :
                weight = datas.get("weight", 1)
                neighborcom = self.statusTab[0].node2com[neighbor]

                if neighborcom not in weights:
                    weights[neighborcom] = np.zeros([len(self.statusTab)])

                # For the graph
                weights[neighborcom][0] = weights[neighborcom][0] + weight

                # For the attributes
                if giniMatrix is not None:
                    weight = giniMatrix[node][neighbor]
                else:
                    weight = -1 * self.dist(self.authorIndex[node], self.authorIndex[neighbor]) / self.nbVertices**2
                weights[neighborcom][1] = weights[neighborcom][1] + weight
        return weights

    def __remove(self, node, com, weight, status) :
        status.degrees[com] = ( status.degrees.get(com, 0.) - status.gdegrees.get(node, 0.) )
        status.internals[com] = float( status.internals.get(com, 0.) - weight - status.loops.get(node, 0.) )
        status.node2com[node] = -1

    def __insert(self, node, com, weight, status) :
        status.node2com[node] = com
        status.degrees[com] = ( status.degrees.get(com, 0.) + status.gdegrees.get(node, 0.) )
        status.internals[com] = float( status.internals.get(com, 0.) + weight + status.loops.get(node, 0.) )

    def __modularity(self, status) :
        links = abs(float(status.total_weight))
        result = 0.
        for community in set(status.node2com.values()) :
            in_degree = abs(status.internals.get(community, 0.))
            degree = abs(status.degrees.get(community, 0.))
            expected = ((degree / (2. * links))**2)
            found = in_degree / links
            if status.total_weight < 0:
                result += expected - found
            else:
                result += found - expected
        return result

class Status :
    """
    To handle several data in one struct.
    Could be replaced by named tuple, but don't want to depend on python 2.6
    """
    node2com = dict([])
    total_weight = 0
    internals = dict([])
    degrees = dict([])
    gdegrees = dict([])
    loops = dict([])

    def __str__(self) :
        return ("------------------------\nnode2com : " + str(self.node2com) + "\n degrees : "
            + str(self.degrees) + "\n gdegrees : "
            + str(self.gdegrees)+ "\n internals : " + str(self.internals)
            + "\n total_weight : " + str(self.total_weight) + "\n loops:"+str(self.loops)+"\n-----------------------")

    def initAttribStatus(self, graph, authorIndex, attributes) :
        """Initialize the status of an attributes list with every node in one community"""
        N = len(graph)
        count = 0

        # Compute the center of gravity using dict
        meanVector = {}
        for v, attrs in attributes.items():
            for attrId, attrValue in attrs.items():
                meanVector[attrId] = meanVector.get(attrId, 0.) + attrValue
        for attrId, attrValue in meanVector.items():
            meanVector[attrId] = meanVector[attrId] / N

        variance = {}
        for node in sorted(graph.nodes()) :
            distanceToCenterOfGravity = 0.
            for attrId, attrValue in meanVector.items():
                variance[attrId] = variance.get(attrId, 0.) + ((attrValue - attributes[authorIndex[node]].get(attrId, 0.)) ** 2)
        inertieTot = 0.
        for v in variance.values():
            inertieTot += (v / N)

        # if(args.verbose):
        #     print "# Total inertia:", inertieTot
        self.total_weight = (0.0  - inertieTot )

        for node in sorted(graph.nodes()) :
            self.node2com[node] = count

            # Compute the distance to the center of gravity
            distanceToCenterOfGravity = 0.
            for attrId, attrValue in meanVector.items():
                distanceToCenterOfGravity += (attrValue - attributes[authorIndex[node]].get(attrId, 0.)) ** 2


            phiHuyghens = -1 * (inertieTot + distanceToCenterOfGravity) / N
            # if(args.verbose):
            #     print "# phiHuyghens(" + str(node) + ") = " + str(phiHuyghens)
            self.degrees[count] = phiHuyghens
            self.gdegrees[node] = phiHuyghens
            self.loops[node] = 0
            self.internals[count] = self.loops[node]
            count = count + 1

    def inducedAttribStatusTab(self, node2com, bijection):
        # if(args.verbose):
        #     print self
        retrobijection = {}
        for k,v in bijection.items():
            retrobijection[v] = k
        self.node2com = dict([])
        oldDegrees = self.degrees
        oldInternals = self.internals

        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.node2com = dict([])
        self.loops = dict([])

        for node in retrobijection:
            self.node2com[node] = node
            deg = oldDegrees[retrobijection[node]]
            self.degrees[node] = deg
            self.gdegrees[node] = deg
            self.loops[node] = oldInternals[retrobijection[node]]
            self.internals[node] = self.loops[node]

    def init(self, graph) :
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight = 'weight')
        for node in sorted(graph.nodes()) :
            self.node2com[node] = count
            deg = float(graph.degree(node, weight = 'weight'))
            self.degrees[count] = deg
            self.gdegrees[node] = deg
            self.loops[node] = float(graph.get_edge_data(node, node, {"weight":0}).get("weight", 1))
            self.internals[count] = self.loops[node]
            count = count + 1

# def loadDataset(path):
#     graph = nx.Graph()
#
#     # Read the graph
#     if(not os.path.isfile(path + ".edgeList")):
#         print "Error: file '" + path + ".edgeList' not found"
#         exit(-1)
#     with open(path + ".edgeList") as f:
#         for line in f.readlines():
#             v1 = int(line.split(" ")[0])
#             v2 = int(line.split(" ")[1])
#             graph.add_node(v1)
#             graph.add_node(v2)
#             graph.add_edge(v1, v2)
#
#     # Read the attributes
#     attributes = {}
#     for n in graph:
#         attributes[n] = {}
#
#     if(not os.path.isfile(path + ".attributes")):
#         print "Error: file '" + path + ".attributes' not found"
#         exit(-1)
#
#     with open(path + ".attributes") as f:
#         for line in f.readlines():
#             vertexId = int(line.split(" ")[0])
#             elems = line.split(" ")[1].split(",")
#             i = 0
#             attrValues = {}
#             for attrValue in elems:
#                 attrValues[i] = float(attrValue)
#                 i = i + 1
#             attributes[vertexId] = attrValues
#
#     # Build authorIndex
#     authorIndex = {}
#     for n in graph:
#         authorIndex[n] = n
#
#     if(args.verbose):
#         print "# Finished reading dataset"
#     if os.path.exists(path + ".2ModLouvain"):
#         os.remove(path + ".2ModLouvain")
#
#     return graph, attributes, authorIndex
#
#
#
# def readToyGraph():
#     graph = nx.Graph()
#     graph.add_node("a")
#     graph.add_node("b")
#     graph.add_node("c")
#     graph.add_node("d")
#     graph.add_node("e")
#     graph.add_edge("a","b")
#     graph.add_edge("b","c")
#     graph.add_edge("c","d")
#     graph.add_edge("d","e")
#     graph.add_edge("a","e")
#     graph.add_edge("b","e")
#     graph.add_edge("c","e")
#     graph.add_edge("b","d")
#     graph.add_edge("a","c")
#     graph.add_edge("a","d")
#
#     authorIndex={}
#     authorIndex["a"]=0
#     authorIndex["b"]=1
#     authorIndex["c"]=2
#     authorIndex["d"]=3
#     authorIndex["e"]=4
#
#     attributes = {
#                 0 : {0 : 2., 1 : 4},
#                 1 : {0 : 8., 1 : 1},
#                 2 : {0 : 7., 1 : 5},
#                 3 : {0 : 12., 1 : 6},
#                 4 : {0 : 1., 1 : 4}}
#     return graph, attributes, authorIndex
#
# def __main() :
#     global args
#     global curDatasetIdx
#     parser = argparse.ArgumentParser(description = 'The 2ModLouvain algorithm', prog = "2ModLouvain")
#     parser.add_argument('dataset', help='Path to the dataset files')
#     parser.add_argument('-verbose', '-v', action='store_true', help='Output algorithm debuging information')
#     parser.add_argument('-toydataset', '-t', action='store_true', help='Use the toy dataset')
#     parser.add_argument('-multipleDataset', '-m', nargs=2, type=int, help='Use the toy dataset' )
    
#     args = parser.parse_args()
    
#     if(args.toydataset):
#         graph, attributes, authorIndex = readToyGraph()
#         algo = ML2(graph, attributes, authorIndex)
#         algo.findPartition()
#     else:
#         if(args.multipleDataset is not None):
#             for i in range(args.multipleDataset[0], args.multipleDataset[1] + 1):
#                 curDatasetIdx = i
#                 graph, attributes, authorIndex = loadDataset(args.dataset + "_" + str(i))
#                 algo = ML2(graph, attributes, authorIndex)
#                 algo.findPartition()
#         else:
#             graph, attributes, authorIndex = loadDataset(args.dataset)
#             algo = ML2(graph, attributes, authorIndex)
#             algo.findPartition()

# if __name__ == "__main__" :
#     __main()
