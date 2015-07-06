#! /usr/bin/python


"""
Lloyd's algorithm
Ref: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/

K-means ++ algorithm By Arthur and Vassilvitskii
Ref: https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

Find optimal K using gap_statistics (http://web.stanford.edu/~hastie/Papers/gap.pdf)
Ref: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/

Alternative approach to find optimal K (http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf)
Ref:  https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/
"""

import numpy as np
import random
import collections
import matplotlib.pyplot as plot
import operator
from data_structure import *
from data_utils import *

# Lloyd's algorithm (standard K-means)
class KMeans():
    def __init__(self, X=None, N=0):
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

    def _init_centroid(self, K):
        self.centroid = random.sample(self.X, K)

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

    def _bounding_box(self):
        X = self.X
        xmin, xmax = min(X, key=lambda a: a[0])[0], max(X, key=lambda a: a[0])[0]
        ymin, ymax = min(X, key=lambda a: a[1])[1], max(X, key=lambda a: a[1])[1]
        return (xmin, xmax), (ymin, ymax)

    def cluster(self, K):
        # Initialize to K random centers
        X = self.X
        self.old_centroid = random.sample(X, K)
        centroid = random.sample(X, K)
        self._init_centroid(K=K)
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
            plot.scatter(x, y, c=color, marker = "o", s = 10, label="Cluster %d" % (cluster_id))
            plot.scatter(centroid_x, centroid_y, c=color, marker = "^", s = 250)
            plot.title("Clustering")

# K-means ++ algorithm
class KPlusPlus(KMeans):
    def _update_dist_from_centroids(self):
        self.D2 = np.array([ min([np.linalg.norm(x-c)**2 for c in self.centroid ]) for x in self.X ])

    def _next_centroid(self):
        self.probs    = self.D2 / self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        rand          = random.random()
        index         = np.where(self.cumprobs >= rand)[0][0]
        return self.X[index]

    def _init_centroid(self, K):
        self.centroid = random.sample(self.X, 1)
        while len(self.centroid) < K:
            self._update_dist_from_centroids()
            self.centroid.append(self._next_centroid())

class OptimalK(KPlusPlus):
    # Alternative approach
    def fk(self, this_k, skm1=0):
        X = self.X
        dimension = len(X[0])
        a = lambda k, dimension: 1 - 3/(4*dimension) if k== 2 \
            else a(k-1, dimension) + (1-a(k-1, dimension)) / 6
        self.cluster(this_k)
        centroid, clusters = self.centroid, self.clusters
        sk = sum([ np.linalg.norm(centroid[i] - point) ** 2 \
                  for i in range(this_k) for point in clusters[i] ])
        if this_k == 1 or skm1 ==0:
            fs = 1
        else:
            fs = sk / (a(this_k, dimension) * skm1)
        return fs, sk

    # Gap statistics method
    def gap(self, this_k):
        X = self.X
        (xmin,xmax), (ymin,ymax) = self._bounding_box()
        self.cluster(K = this_k)
        centroid, clusters = self.centroid, self.clusters
        Wk = np.log(sum([np.linalg.norm(centroid[cluster_id]-c)**2/(2*len(c))  for cluster_id in range(this_k) for c in clusters[cluster_id]]))
        # Create B reference datasets
        B = 10
        BWkbs = [0] * B
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax), random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            kb = OptimalK(X=Xb)
            kb.cluster(K = this_k)
            ms, cs = kb.centroid, kb.clusters
            BWkbs[i] = np.log(sum([np.linalg.norm(ms[j]-c)**2/(2*len(c)) for j in range(this_k) for c in cs[j]]))

        Wkb = sum(BWkbs)/B
        sk = np.sqrt(sum((BWkbs-Wkb)**2)/float(B))*np.sqrt(1+1/B)
        return Wk, Wkb, sk

    def run(self, max_k, algorithm='both'):
        ks = range(1,max_k)
        fs = [0] * len(ks)
        Wks, Wkbs, sks = [0] * (len(ks)+1), [0] * (len(ks)+1), [0] * (len(ks)+1)
        # Special case K=1
        self._init_centroid(K=1)
        if algorithm == 'f':
            fs[0], Sk = self.fK(K=1)
        elif algorithm == 'gap':
            Wks[0], Wkbs[0], sks[0] = self.gap(1)
        else:
            fs[0], Sk = self.fk(1)
            Wks[0], Wkbs[0], sks[0] = self.gap(1)
        # Rest of Ks
        for k in ks[1:]:
            self._init_centroid(K=k)
            if algorithm == 'f':
                fs[k-1], Sk = self.fk(K=k, skm1=Sk)
            elif algorithm == 'gap':
                Wks[k-1], Wkbs[k-1], sks[k-1] = self.gap(k)
            else:
                fs[k-1], Sk = self.fk(k, skm1=Sk)
                Wks[k-1], Wkbs[k-1], sks[k-1] = self.gap(k)
        if algorithm == 'f':
            self.fs = fs
            self.fk_optimal = np.where(self.fs == min(self.fs))[0][0] + 1
        elif algorithm == 'gap':
            G = []
            for i in range(len(ks)):
                G.append((Wkbs[i]-Wks[i]) - ((Wkbs[i+1]-Wks[i+1]) - sks[i+1]))
            self.G = np.array(G)
            self.gap_optimal = np.where(self.G > 0)[0][0] + 1
        else:
            self.fs = fs
            self.fk_optimal = np.where(self.fs == min(self.fs))[0][0] + 1
            G = []
            for i in range(len(ks)):
                G.append((Wkbs[i]-Wks[i]) - ((Wkbs[i+1]-Wks[i+1]) - sks[i+1]))
            self.G = np.array(G)
            self.gap_optimal = np.where(self.G > 0)[0][0] + 1

    def visualize(self):
        X = self.X
        ks = range(1, len(self.fs)+1)
        fig = plot.figure(figsize=(18,5))
        # Plot 1
        ax1 = fig.add_subplot(131)
        ax1.set_xlim(-1,1)
        ax1.set_ylim(-1,1)
        ax1.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
        title1 = 'N=%s' % (str(len(X)))
        ax1.set_title(title1, fontsize=16)
        # Plot 2
        ax2 = fig.add_subplot(132)
        ax2.set_ylim(0, 1.25)
        ax2.plot(ks, self.fs, 'ro-', alpha=0.6)
        ax2.set_xlabel('Number of clusters K', fontsize=16)
        ax2.set_ylabel('f(K)', fontsize=16)
        title2 = 'f(K) finds %s clusters' % (self.fk_optimal)
        ax2.set_title(title2, fontsize=16)
        # Plot 3
        ax3 = fig.add_subplot(133)
        ax3.bar(ks, self.G, alpha=0.5, color='g', align='center')
        ax3.set_xlabel('Number of clusters K', fontsize=16)
        ax3.set_ylabel('Gap', fontsize=16)
        title3 = 'Gap statistic finds %s clusters' % (self.gap_optimal)
        ax3.set_title(title3, fontsize=16)
        ax3.xaxis.set_ticks(range(1,len(ks)+1))
        plot.savefig('optimalK.png', bbox_inches='tight', dpi=100)


def main():
    N = 100
    K = 3
    X = init_board_gauss(N, K)
    """
    k_means = KMeans(X=X)
    k_means.cluster(K)
    k_means.visualize("K-means, Sample size: %d, cluters: %d" % (N, K))
    plot.show()

    """
    optimal_k = OptimalK(X = X)
    optimal_k.run(max_k = 2 * K)

    k_pp = KPlusPlus(X=X)
    k_pp.cluster(K=optimal_k.gap_optimal)
    k_pp.visualize("K-means ++, Sample size: %d, cluters: %d" % (N, optimal_k.gap_optimal))
    plot.show()

if __name__ == "__main__":
    main()
