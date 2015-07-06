#! /usr/bin/python

import math
from data_structure import *

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
            p.rd = None
            p.processed = False
        self.unprocessed = [p for p in self.points]
        self.ordered = []

    # --------------------------------------------------------------------------
    # distance from a point to its nth neighbor (n = min_cluser_size)
    # --------------------------------------------------------------------------
    def _core_distance(self, point, neighbors):
        if point.cd is not None: return point.cd
        if len(neighbors) >= self.min_cluster_size - 1:
            sorted_neighbors = sorted([n.distance(point) for n in neighbors])
            point.cd = sorted_neighbors[self.min_cluster_size - 2]
            return point.cd

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
        point.processed = True
        self.unprocessed.remove(point)
        self.ordered.append(point)

    # --------------------------------------------------------------------------
    # update seeds if a smaller reachability distance is found
    # --------------------------------------------------------------------------
    def _update(self, neighbors, point, seeds):
        # for each of point's unprocessed neighbors n...
        for n in [n for n in neighbors if not n.processed]:

            # find new reachability distance new_rd
            # if rd is null, keep new_rd and add n to the seed list
            # otherwise if new_rd < old rd, update rd
            new_rd = max(point.cd, point.distance(n))
            if n.rd is None:
                n.rd = new_rd
                seeds.append(n)
            elif new_rd < n.rd:
                n.rd = new_rd

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

                    seeds.sort(key=lambda n: n.rd)
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

        clusters = []
        separators = []

        for i in range(len(self.ordered)):
            this_i = i
            next_i = i + 1
            this_p = self.ordered[i]
            this_rd = this_p.rd if this_p.rd else float('infinity')

            # use an upper limit to separate the clusters

            if this_rd > cluster_threshold:
                separators.append(this_i)

        separators.append(len(self.ordered))

        for i in range(len(separators) - 1):
            start = separators[i]
            end = separators[i + 1]
            if end - start >= self.min_cluster_size:
                clusters.append(Cluster(self.ordered[start:end]))

        return clusters


def main():
    # LOAD SOME POINTS
    points = [
        Point(37.769006, -122.429299), # cluster #1
        Point(37.769044, -122.429130), # cluster #1
        Point(37.768775, -122.429092), # cluster #1
        Point(37.776299, -122.424249), # cluster #2
        Point(37.776265, -122.424657), # cluster #2
    ]

    optics = Optics(points, 100, 2) # 100m radius for neighbor consideration, cluster size >= 2 points
    optics.run()                    # run the algorithm
    clusters = optics.cluster(50)   # 50m threshold for clustering

    for cluster in clusters:
        print cluster.points


if __name__ == "__main__":
    main()
