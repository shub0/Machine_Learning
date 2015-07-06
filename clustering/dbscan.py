#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import operator
import collections
from data_utils import *
from data_structure import *

# A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu
# dbscan: density based spatial clustering of applications with noise
# Ref: https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
#      https://en.wikipedia.org/wiki/DBSCAN

################################################################################
# Implementation of Density Based Spatial Clustering of Applications with Noise
################################################################################
NOISE = 0
class Dbscan(ClusteringAlgorithm):
    def __init__(self, points, min_cluster_size, eps):
        self.points = points
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.clusters = [Cluster()]

    # neighbours for points within eps radius
    def _region_query(self, point):
        return set([p for p in self.points if p.distance(point) < self.eps])

    # expand current cluster with new points
    def _expand_cluster(self, point, cluster, seeds):
        cluster.append(point)
        while len(seeds) > 0:
            curr_point = seeds.pop()
            if not curr_point.processed:
               curr_point.process()
               neighbours = self._region_query(curr_point)
               if len(neighbours) >= self.min_cluster_size:
                   seeds.update(neighbours)
            if not curr_point.clustered:
                cluster.append(curr_point)

    # Clustering points
    # DBSCAN algorithm does not require a give K
    def run(self, K = None):
        cluster = Cluster()
        for point in self.points:
            if point.processed:
                continue
            point.process()
            seeds = self._region_query(point)
            if len(seeds) >= self.min_cluster_size:
                new_cluster = Cluster()
                self._expand_cluster(point, new_cluster, seeds)
                self.clusters.append(new_cluster)

        for point in self.points:
            if not point.clustered:
                self.clusters[NOISE].append(point)


if __name__ == "__main__":
    N = 100
    data = init_board_half_moon(N=N, offset = 0.5)
    points = [ Point(point[0], point[1]) for point in data ]
    eps = 5
    min_points = N / 5
    db = Dbscan(points, min_points, eps)
    db.run()
    visualize(db.clusters, "DBSCAN Alg")
