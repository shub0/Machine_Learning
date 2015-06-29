#! /usr/bin/python

import numpy as np
import random
import collections
import matplotlib.pyplot as plot

"""
Lloyd's algorithm
Ref: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
"""

def cluster_points(X, mu):
    clusters  = collections.defaultdict(list)
    for x in X:
        cluster_index = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
        clusters[cluster_index].append(x)
    return clusters

def evaluate_centers(centroid, clusters):
    new_centroid = list()
    cluster_ids = sorted(clusters.keys())
    for cluster_id in cluster_ids:
        new_centroid.append(np.mean(clusters[cluster_id], axis = 0))
    return new_centroid

def has_converged(new_centroid, old_centroid):
    return (set([tuple(a) for a in new_centroid]) == set([tuple(a) for a in old_centroid]))

def find_centers(X, K):
    # Initialize to K random centers
    old_centroid = random.sample(X, K)
    centroid = random.sample(X, K)
    while not has_converged(centroid, old_centroid):
        old_centroid = centroid
        # Assign all points in X to clusters
        clusters = cluster_points(X, centroid)
        # Reevaluate centers
        centroid = evaluate_centers(old_centroid, clusters)
    return(centroid, clusters)

def visualize(clusters, msg = None):
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    if msg:
        plot.title(msg)
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
    centroid, clusters = find_centers(X, K)
    visualize(clusters, "Sample size: %d, cluters: %d" % (N, K))
    plot.show()

if __name__ == "__main__":
    main()
