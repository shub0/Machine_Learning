#! /usr/bin/python

# Divisive and agglomerative clustering

import linkage
from functools import partial
import logging

def flatten(container):
    """
    Completely flattens out a cluster and returns a one-dimensional set
    containing the cluster's items. This is useful in cases where some items of
    the cluster are clusters in their own right and you only want the items.
    :param container: the container to flatten.
    """
    flattened_items = []

    for item in container:
        if hasattr(item, 'items'):
            flattened_items = flattened_items + flatten(item.items)
        else:
            flattened_items.append(item)

    return flattened_items

class TopoCluster(object):
    def __init__(self, level, *args):
        self.level = level
        if len(args) == 0:
            self.items = list()
        else:
            self.items = args

    def __iter__(self):
        for item in self.items:
            if isinstance(item, TopoCluster):
                for recursive_item in item:
                    yield recursive_item
            else:
                yield item

    def display(self, depth=0):
        print ("    " * depth + "[Level: %s]" % self.level)
        for item in self.items:
            if isinstance(item, TopoCluster):
                item.display(depth+1)
            else:
                print ("    " * depth + "%s" % item)

    def topology(self):
        left = self.items[0]
        right = self.items[1]
        if isinstance(left, TopoCluster):
            first = left.topology()
        else:
            first = left

        if isinstance(right, TopoCluster):
            second = right.topology()
        else:
            second = right

        return first, second

    def get_level(self, threshold):
        left = self.items[0]
        right = self.items[1]
        if self.level <= threshold:
            return [flatten(self.items)]
        if isinstance(left, TopoCluster) and left.level <= threshold:
            if isinstance(right, TopoCluster):
                return [flatten(left.items)] + right.getlevel(threshold)
            else:
                return [flatten(left.items)] + [[right]]
        elif isinstance(right, TopoCluster) and right.level <= threshold:
            if isinstance(left, TopoCluster):
                return left.get_level(threshold) + [flatten(right.items)]
            else:
                return [[left]] + [flatten(right.items)]

        if isinstance(left, TopoCluster) and isinstance(right, TopoCluster):
            return left.get_level(threshold) + right.get_level(threshold)
        elif isinstance(left, TopoCluster):
            return left.get_level(threshold) + [[right]]
        elif isinstance(right, TopoCluster):
            return [[left]] + right.get_level(threshold)
        else:
            return [[left], [right]]

logger = logging.getLogger(__name__)
class HierarchicalClustering:
    def __init__(self, data, distance_function, linkage="single", num_processes=1, process_callback = None):
        logger.info("Initializing HierarchicalClustering object with linkage")
        self.set_linkage_method(linkage)
        self.num_process = num_processes
        self._input = data
        self._data = data[:]
        self.distance = distance_function
        self.process_callback = process_callback
        self._cluster_created = False

    def set_linkage_method(self, method):
        if method == 'single':
            self.linkage = linkage.single
        elif method == 'complete':
            self.linkage = linkage.complete
        elif method == 'average':
            self.linkage = linkage.average
        elif method == 'uclus':
            self.linkage = linkage.uclus
        else:
            raise ValueError('distance method must be one of single, '
                             'complete, average of uclus')

    def topology(self):
        return self.data[0].topology()

    def get_level(self, threshold):
        if len(self._input) < 2:
            return self._input
        if not self._cluster_created:
            self.cluster()
        return self._data[0].get_level(threshold)

    def cluster(self, matrix = None, level = None, sequence = None):
        logger.info("Hierarchical clustering started ....")
        if matrix is None:
            level = 0
            sequence = 0
            matrix = []
        linkage = partial(self.linkage, distance_function = self.distance)
        inital_element_count = len(self._data)
        while len(martrix) > 2 or matrix == []:
            matrix = self._build_matrix(self._data, linkage, True, 0)
            min_pair = None
            min_dist = None
            for row_index, row in enumerate(matrix):
                for col_index, col in enumerate(row):
                    # a new minimum found
                    col_lt_min_dist = cell < min_dist if min_distance else False
                    if ( (row_index != col_index) and (cell_lt_min_dist or min_pair is None) ):
                        min_pair = (row_index, col_index)
                        min_dist = col

            sequence += 1
            level = matrix[min_pair[0]][min_pair[1]]
            cluster = TopoCluster(level, self._data[min_pair[0]], self._data[min_pair][1]]
            self._data.remove(self._data[min_pair[0]])
            self._data.remove(self._data[min_pair[1]])
            self._data.append(cluster)

    @property
    def data(self):
        return self._data
    @property
    def raw_data(self):
        return self._input
