#! /usr/bin/python

import numpy as np
import random
import collections
import matplotlib.pyplot as plot

"""
Lloyd's algorithm
Ref: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
K-means ++ algorithm By Arthur and Vassilvitskii
"""

# Lloyd's algorithm (standard K-means)
class KMeans():
    def __init__(self, K, X=None, N=0):
        self.K = K
        if X == None:
            if N == 0:
                raise ValueError("Data size must positive")
            self.N = N
            self.X = init_board_gauss(N, K)
        else:
            self.X = X
            self.N = len(X)
        self.centroid = None
        self.clusters = None

    def init_centroid(self):
        print "Standard Init"
        self.centroid = random.sample(self.X, self.K)

    def _cluster_points(self):
        mu = self.centroid
        clusters  = collections.defaultdict(list)
        for x in self.X:
            cluster_index = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
            clusters[cluster_index].append(x)
        self.clusters = clusters

    def _evaluate_centers(self):
        clusters = self.clusters
        new_centroid = list()
        cluster_ids = sorted(clusters.keys())
        for cluster_id in cluster_ids:
            new_centroid.append(np.mean(clusters[cluster_id], axis = 0))
        self.centroid = new_centroid


    def _has_converged(self):
        K = len(self.centroid)
        return(set([tuple(a) for a in self.centroid]) ==  set([tuple(a) for a in self.old_centroid])  and len(set([tuple(a) for a in self.centroid])) == K)

    def find_centroids(self):
        # Initialize to K random centers
        X = self.X
        K = self.K
        self.old_centroid = random.sample(X, K)
        centroid = random.sample(X, K)
        self.init_centroid()
        while not self._has_converged():
            self.old_centroid = self.centroid
            # Assign all points in X to clusters
            self._cluster_points()
            # Reevaluate centers
            centroid = self._evaluate_centers()

    def visualize(self, msg = None):
        clusters = self.clusters
        colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
        if msg:
            plot.title(msg)
        plot.xlim(-1, 1)
        plot.ylim(-1, 1)
        for cluster_id in clusters.keys():
            if cluster_id < len(colors):
                color = colors[cluster_id]
            else:
                color = "black"
            x = [ point[0] for point in clusters[cluster_id] ]
            y = [ point[1] for point in clusters[cluster_id] ]
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)
            plot.scatter(x, y, c=color, marker = "o", s = 10)
            plot.scatter(centroid_x, centroid_y, c=color, marker = "^", s = 250)

class KPlusPlus(KMeans):
    def _update_dist_from_centroids(self):
        self.D2 = np.array([ min([np.linalg.norm(x-c)**2 for c in self.centroid ]) for x in self.X ])

    def _next_centroid(self):
        self.probs    = self.D2 / self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        rand          = random.random()
        index         = np.where(self.cumprobs >= rand)[0][0]
        return self.X[index]

    def init_centroid(self):
        print "Initialize with K++"
        self.centroid = random.sample(self.X, 1)
        while len(self.centroid) < self.K:
            self._update_dist_from_centroids()
            self.centroid.append(self._next_centroid())


def init_board(N):
    X = np.array([ (random.uniform(-1, 1), random.uniform(-1,1)) for i in range(N) ])
    return X

def init_board_gauss(N, K):
    n = float(N) / K
    X = list()
    for index in range(K):
        mu    = (random.uniform(-1, 1), random.uniform(-1, 1))
        sigma = random.uniform(0.05, 0.5)
        x = list()
        while len(x) < n:
            a, b = np.array([ np.random.normal(mu[0], sigma), np.random.normal(mu[1], sigma) ])
            if abs(a) < 1 and abs(b) < 1:
                x.append([a, b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

def main():
    N = 1000
    K = 3
    X = init_board_gauss(N, K)

    k_means = KMeans(K=K, X=X)
    k_means.find_centroids()
    k_means.visualize("K-means, Sample size: %d, cluters: %d" % (N, K))
    plot.show()

    k_pp = KPlusPlus(K=K, X=X)
    k_pp.find_centroids()
    k_pp.visualize("K-means ++, Sample size: %d, cluters: %d" % (N, K))
    plot.show()

if __name__ == "__main__":
    main()
