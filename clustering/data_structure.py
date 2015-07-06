import math
import numpy as np

################################################################################
# POINT
################################################################################

class Point:
    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.cd = None              # core distance
        self.rd = None              # reachability distance
        self.processed = False      # has this point been processed?
        self.clustered = False      # has this point been clustered?

    # --------------------------------------------------------------------------
    # calculate the distance between any two points on earth
    # --------------------------------------------------------------------------

    def distance(self, point):
        return math.sqrt( (self.x - point.x) ** 2 + (self.y - point.y) ** 2 )

    # --------------------------------------------------------------------------
    # Mark point as processed
    # --------------------------------------------------------------------------
    def process(self):
        self.processed = True

    # --------------------------------------------------------------------------
    # Mark point as clustered
    # --------------------------------------------------------------------------
    def cluster(self):
        self.clustered = True

    def __repr__(self):
        return '(%f, %f)' % (self.x, self.y)

################################################################################
# CLUSTER
################################################################################
class Cluster:

    def __init__(self, points = list()):
        self.points = list(points)

    # --------------------------------------------------------------------------
    # Append a new point to the cluster
    # --------------------------------------------------------------------------
    def append(self, point):
        point.cluster()
        self.points.append(point)

    # --------------------------------------------------------------------------
    # calculate the centroid for the cluster
    # --------------------------------------------------------------------------
    def centroid(self):
        return Point(sum([p.x for p in self.points])/len(self.points),
            sum([p.y for p in self.points])/len(self.points))

    # --------------------------------------------------------------------------
    # calculate the region (centroid, bounding radius) for the cluster
    # --------------------------------------------------------------------------
    def region(self):
        centroid = self.centroid()
        radius = reduce(lambda r, p: max(r, p.distance(centroid)), self.points)
        return centroid, radius



################################################################################
# Interface for all clustering algorithm
################################################################################
class ClusteringAlgorithm:
    def cluster(self, K):
        raise NotImplementedError()
