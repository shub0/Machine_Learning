#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import matplotlib.pyplot as plot
import operator
import collections
from data_generator import *
from sklearn.cluster import DBSCAN

UNCLASSIFIED = False
NOISE = -1

# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise
# Ref: https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
#      https://en.wikipedia.org/wiki/DBSCAN

def _dist(p, q):
    return math.sqrt(np.power(np.array(p)-np.array(q), 2).sum())

def _eps_neighborhood(p, q, eps):
    return _dist(p, q) < eps

def _region_query(m, point_id, eps):
    n_points = len(m)
    seeds = list()
    for id in range(n_points):
        if _eps_neighborhood(m[point_id], m[id], eps):
            seeds.append(id)
    return seeds

def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for id in seeds:
            classifications[id] = cluster_id

        while len(seeds) > 0:
            current_point = seeds.pop(0)
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for id in range(len(results)):
                    result_point = results[id]
                    if classifications[result_point] == UNCLASSIFIED:
                        seeds.append(result_point)
                        classifications[result_point] = cluster_id
                    if classifications[result_point] == NOISE:
                        classifications[result_point] = cluster_id
        return True

def dbscan(data, eps, min_points):
    """
    Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN

    scikit-learn probably has a better implementation

    Uses Euclidean Distance as the measure

    Inputs:
    data - a array of tuple (locations)
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster

    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """

    cluster_id = 0
    n_points = len(data)
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(n_points):
        point = data[point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
                print "Cluster %d done ...." % (cluster_id)
                cluster_id += 1

    clusters = collections.defaultdict(list)
    for index in range(n_points):
        clusters[classifications[index]].append(data[index])
    return clusters

def visualize(data):
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    for cluster_id in data.keys():
        x = [ point[0] for point in data[cluster_id] ]
        y = [ point[1] for point in data[cluster_id] ]
        if cluster_id == NOISE:
            plot.scatter(x, y, c='black', label="Noise")
        else:
            plot.scatter(x, y, c=colors[cluster_id], label="Cluster %d" % (cluster_id))
    plot.legend()
    plot.show()

def sklearn_demo(data, eps, min_points):
    N = len(data)
    db = DBSCAN(eps = eps, min_samples = min_points).fit(data)
    clusters = collections.defaultdict(list)
    for index in range(N):
        clusters[db.labels_[index]].append(data[index])
    return clusters


if __name__ == "__main__":
    N = 1000
    data = init_board_half_moon(N=N, offset = 0.5)
    eps = 0.5
    min_points = N / 5
    visualize(dbscan(data, eps, min_points))
    visualize(sklearn_demo(data, eps, min_points))
