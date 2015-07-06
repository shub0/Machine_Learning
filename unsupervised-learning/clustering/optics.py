#! /usr/bin/python

import math
from data_structure import *
from data_utils import *

################################################################################
# OPTICS
################################################################################
class Optics:
    def __init__(self, points, max_radius, min_cluster_size):
        self.points = points
        self.max_radius = max_radius                # maximum radius to consider
        self.min_cluster_size = min_cluster_size    # minimum points in cluster

    # --------------------------------------------------------------------------
    # get ready for a clustering run
    # --------------------------------------------------------------------------
    def _setup(self):
        for p in self.points:
            p.reachability_distance = None
            p.processed = False
        self.unprocessed = [p for p in self.points]
        self.ordered = []

    # --------------------------------------------------------------------------
    # distance from a point to its nth neighbor (n = min_cluser_size)
    # --------------------------------------------------------------------------
    def _core_distance(self, point, neighbors):
        if point.core_distance is not None:
            return point.core_distance
        if len(neighbors) > self.min_cluster_size:
            sorted_neighbors = sorted([n.distance(point) for n in neighbors])
            point.core_distance = sorted_neighbors[self.min_cluster_size - 1]
            return point.core_distance
        return None

    # --------------------------------------------------------------------------
    # neighbors for a point within max_radius
    # --------------------------------------------------------------------------
    def _neighbors(self, point):

        return [p for p in self.points if p is not point and
            p.distance(point) <= self.max_radius]

    # --------------------------------------------------------------------------
    # mark a point as processed
    # --------------------------------------------------------------------------
    def _processed(self, point):
        point.process()
        self.unprocessed.remove(point)
        self.ordered.append(point)

    # --------------------------------------------------------------------------
    # update seeds if a smaller reachability distance is found
    # --------------------------------------------------------------------------
    def _update(self, neighbors, point, seeds):
        # for each of point's unprocessed neighbors n...
        for n in neighbors:
            if n.processed:
                continue
            # find new reachability distance new_reachability_distance
            # if reachability_distance is null, keep new_reachability_distance and add n to the seed list
            # otherwise if new_reachability_distance < old reachability_distance, update reachability_distance
            new_reachability_distance = max(point.core_distance, point.distance(n))
            if n.reachability_distance is None:
                n.reachability_distance = new_reachability_distance
                seeds.append(n)
            elif new_reachability_distance < n.reachability_distance:
                n.reachability_distance = new_reachability_distance

    # --------------------------------------------------------------------------
    # run the OPTICS algorithm
    # --------------------------------------------------------------------------
    def run(self):
        self._setup()
        # for each unprocessed point (p)...
        while self.unprocessed:
            point = self.unprocessed[0]

            # mark p as processed
            # find p's neighbors
            self._processed(point)
            point_neighbors = self._neighbors(point)

            # if p has a core_distance, i.e has min_cluster_size - 1 neighbors
            if self._core_distance(point, point_neighbors) is not None:

                # update reachability_distance for each unprocessed neighbor
                seeds = []
                self._update(point_neighbors, point, seeds)

                # as long as we have unprocessed neighbors...
                while(seeds):

                    # find the neighbor n with smallest reachability distance
                    seeds.sort(key=lambda n: n.reachability_distance)
                    n = seeds.pop(0)

                    # mark n as processed
                    # find n's neighbors
                    self._processed(n)
                    n_neighbors = self._neighbors(n)
                    # if p has a core_distance...
                    if self._core_distance(n, n_neighbors) is not None:
                        # update reachability_distance for each of n's neighbors
                        self._update(n_neighbors, n, seeds)

        # when all points have been processed
        # return the ordered list
        return self.ordered

    # --------------------------------------------------------------------------
    def cluster(self, cluster_threshold):

        self.clusters = list()
        separators = list()

        for i in range(len(self.ordered)):
            this_i = i
            next_i = i + 1
            this_p = self.ordered[i]
            this_reachability_distance = this_p.reachability_distance if this_p.reachability_distance else float('infinity')

            # use an upper limit to separate the clusters

            if this_reachability_distance > cluster_threshold:
                separators.append(this_i)

        separators.append(len(self.ordered))

        for i in range(len(separators) - 1):
            start = separators[i]
            end = separators[i + 1]
            if end - start >= self.min_cluster_size:
                self.clusters.append(Cluster(self.ordered[start:end]))


def main():
    # LOAD SOME POINTS
    N = 1000
    data = init_board_gauss(N=N, K=3)
    points = [ Point(point[0], point[1]) for point in data ]
    eps = .2
    min_points = N / 5
    visualize([Cluster(points)], "Orig Data")

    optics = Optics(points, eps, min_points)
    optics.run()                    # run the algorithm
    optics.cluster(eps)
    visualize(optics.clusters, "Optics")


if __name__ == "__main__":
    main()
