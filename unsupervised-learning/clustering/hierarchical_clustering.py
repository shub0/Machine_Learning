#! /usr/bin/python

# Divisive and agglomerative clustering

import linkage
from functools import partial
import logging
import unittest
from sys import hexversion
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)
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
                return [flatten(left.items)] + right.get_level(threshold)
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

class HierarchicalClustering:
    def __init__(self, data, distance_function, linkage="single"):
        logger.info("Initializing HierarchicalClustering object with linkage")
        self.set_linkage_method(linkage)
        self._input = data
        self._data = data[:]
        self.distance = distance_function
        self._cluster_created = False

    def set_linkage_method(self, method):
        if method == "single":
            self.linkage = linkage.single
        elif method == "complete":
            self.linkage = linkage.complete
        elif method == "average":
            self.linkage = linkage.average
        elif method == "uclus":
            self.linkage = linkage.uclus
        else:
            raise ValueError("distance method must be one of single, "
                             "complete, average of uclus")

    @staticmethod
    def build_matrix(data, linkage_func, symmetric=False, diagonal=None):
        matrix = list()
        for row_index, row_item in enumerate(data):
            logger.debug( "Generating row %s/%s (%.2f%%)", row_index, len(data), 100.0 * row_index / len(data) )
            row = dict()
            for col_index, col_item in enumerate(data):
                if diagonal is not None and col_index == row_index:
                    row[col_index] = diagonal
                elif symmetric and col_index < row_index:
                    pass
                else:
                    if not hasattr(row_item, "__iter__"):
                        row_item = [row_item]
                    if not hasattr(col_item, "__iter__"):
                        col_item = [col_item]
                    row[col_index] = linkage_func(row_item, col_item)

            if symmetric:
                for col_index, col_item in enumerate(data):
                    if col_index >= row_index:
                        break
                    row[col_index] = matrix[col_index][row_index]
            row_indexed = [ row[index] for index in range(len(data)) ]
            matrix.append(row_indexed)
        logger.info("Matrix generation completed ...")
        return matrix

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

        linkage_func = partial(self.linkage, distance_function = self.distance)
        inital_element_count = len(self._data)
        while len(matrix) > 2 or matrix == []:
            matrix = HierarchicalClustering.build_matrix(self._data, linkage_func, True, 0)
            min_pair = None
            min_distance = None

            for row_index, row in enumerate(matrix):
                for col_index, col in enumerate(row):
                    # skip diagnonal item
                    if row_index == col_index:
                        continue
                    # a new minimum found
                    if min_pair is None or col < min_distance:
                        min_pair = (row_index, col_index)
                        min_distance = col

            sequence += 1
            level = min_distance
            # combine two clusters
            min_index = min(min_pair)
            max_index = max(min_pair)
            cluster = TopoCluster(level, self._data[min_pair[0]], self._data[min_pair[1]])
            # remove old clusters
            del self._data[max_index]
            del self._data[min_index]
            # append new agglomerative cluster
            self._data.append(cluster)

        self._cluster_created = True
        logger.info("Clustering completed")

    @property
    def data(self):
        return self._data

    @property
    def raw_data(self):
        return self._input

class Py23TestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Py23TestCase, self).__init__(*args, **kwargs)
        if hexversion < 0x030000f0:
            self.assertCItemsEqual = self.assertItemsEqual
        else:
            self.assertCItemsEqual = self.assertCountEqual

class HClusterSmallListTestCase(Py23TestCase):
    def testCluster(self):
        """
        Testing if hierarchical clustering a set of length 1 returns a set of
        length 1
        """
        cl = HierarchicalClustering([876], lambda x, y: abs(x - y))
        self.assertCItemsEqual([876], cl.get_level(40))

    def testEmptyCluster(self):
        """
        Testing if hierarchical clustering an empty list returns an empty list
        """
        cl = HierarchicalClustering([], lambda x, y: abs(x - y))
        self.assertEqual([], cl.get_level(40))


class HClusterIntegerTestCase(Py23TestCase):

    def __init__(self, *args, **kwargs):
        super(HClusterIntegerTestCase, self).__init__(*args, **kwargs)
        if hexversion < 0x030000f0:
            self.assertCItemsEqual = self.assertItemsEqual
        else:
            self.assertCItemsEqual = self.assertCountEqual

    def setUp(self):
        self.__data = [791, 956, 676, 124, 564, 84, 24, 365, 594, 940, 398,
                       971, 131, 365, 542, 336, 518, 835, 134, 391]

    def testSingleLinkage(self):
        "Basic Hierarchical Clustering test with integers"
        cl = HierarchicalClustering(self.__data, lambda x, y: abs(x - y))
        result = cl.get_level(40)
        # sort the values to make the tests less prone to algorithm changes
        result = [sorted(_) for _ in result]
        self.assertCItemsEqual([
            [24],
            [336, 365, 365, 391, 398],
            [518, 542, 564, 594],
            [676],
            [791],
            [835],
            [84, 124, 131, 134],
            [940, 956, 971],
        ], result)

    def testCompleteLinkage(self):
        "Basic Hierarchical Clustering test with integers"
        cl = HierarchicalClustering(self.__data,
                                    lambda x, y: abs(x - y),
                                    linkage="complete")
        result = cl.get_level(40)

        # sort the values to make the tests less prone to algorithm changes
        result = sorted([sorted(_) for _ in result])

        expected = [
            [24],
            [84],
            [124, 131, 134],
            [336, 365, 365],
            [391, 398],
            [518],
            [542, 564],
            [594],
            [676],
            [791],
            [835],
            [940, 956, 971],
        ]
        self.assertEqual(result, expected)

    def testUCLUS(self):
        "Basic Hierarchical Clustering test with integers"
        cl = HierarchicalClustering(self.__data,
                                    lambda x, y: abs(x - y),
                                    linkage="uclus")
        expected = [
            [24],
            [84],
            [124, 131, 134],
            [336, 365, 365, 391, 398],
            [518, 542, 564],
            [594],
            [676],
            [791],
            [835],
            [940, 956, 971],
        ]
        result = sorted([sorted(_) for _ in cl.get_level(40)])
        self.assertEqual(result, expected)

    def testAverageLinkage(self):
        cl = HierarchicalClustering(self.__data,
                                    lambda x, y: abs(x - y),
                                    linkage="average")
        # TODO: The current test-data does not really trigger a difference
        # between UCLUS and "average" linkage.
        expected = [
            [24],
            [84],
            [124, 131, 134],
            [336, 365, 365, 391, 398],
            [518, 542, 564],
            [594],
            [676],
            [791],
            [835],
            [940, 956, 971],
        ]
        result = sorted([sorted(_) for _ in cl.get_level(40)])
        self.assertEqual(result, expected)

    def testRawData(self):
        cl = HierarchicalClustering(self.__data, lambda x, y: abs(x - y))
        new_data = []
        [new_data.extend(_) for _ in cl.get_level(40)]
        self.assertEqual(sorted(new_data), sorted(self.__data))

class HClusterStringTestCase(Py23TestCase):
    def sim(self, x, y):
        sm = SequenceMatcher(lambda x: x in ". -", x, y)
        return 1 - sm.ratio()

    def setUp(self):
        self.__data = ("Lorem ipsum dolor sit amet consectetuer adipiscing "
                       "elit Ut elit Phasellus consequat ultricies mi Sed "
                       "congue leo at neque Nullam").split()

    def testDataTypes(self):
        "Test for bug #?"
        cl = HierarchicalClustering(self.__data, self.sim)
        for item in cl.get_level(0.5):
            self.assertEqual(
                type(item), type([]),
                "Every item should be a list!")

    def testCluster(self):
        "Basic Hierachical clustering test with strings"
        self.skipTest("These values lead to non-deterministic results. "
                      "This makes it untestable!")
        cl = HierarchicalClustering(self.__data, self.sim)
        self.assertEqual([
            ["ultricies"],
            ["Sed"],
            ["Phasellus"],
            ["mi"],
            ["Nullam"],
            ["sit", "elit", "elit", "Ut", "amet", "at"],
            ["leo", "Lorem", "dolor"],
            ["congue", "neque", "consectetuer", "consequat"],
            ["adipiscing"],
            ["ipsum"],
        ], cl.get_level(0.5))

    def testUnmodifiedData(self):
        cl = HierarchicalClustering(self.__data, self.sim)
        new_data = []
        [new_data.extend(_) for _ in cl.get_level(0.5)]
        self.assertEqual(sorted(new_data), sorted(self.__data))


if __name__ == "__main__":
    suite = unittest.TestSuite((
        unittest.makeSuite(HClusterIntegerTestCase),
        unittest.makeSuite(HClusterSmallListTestCase),
        unittest.makeSuite(HClusterStringTestCase),
    ))
    unittest.TextTestRunner(verbosity=2).run(suite)
