#! /usr/bin/python
import sys
import time
from optparse import OptionParser
from itertools import imap
from collections import defaultdict
import logging

class FPNode:
    def __init__(self, item, count = 1):
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add_child(self, child):
        """Add new child to node"""
        if not isinstance(child, FPNode):
            raise TypeError("Can only add FPNode as child")
        if not child.item in self._children:
            self._children[child.item] = child
            child.parent = self

    def search_child(self, item):
        return self._children.get(item, None)

    def remove_child(self, child):
        try:
            if self._children[child.item] is child:
                del self._children[child.item]
                child.parent = None
                for sub_child in child.children:
                    try:
                        # Merger case: we already have a child for that item, so
                        # add the sub-child's count to our child's count.
                        self._children[sub_child.item]._count += sub_child.count
                        sub_child.parent = None # it's an orphan now
                    except KeyError:
                        # Turns out we don't actually have a child, so just add
                        # the sub-child as our own child.
                        self.add_child(sub_child)
                child._children = {}
            else:
                raise ValueError("given child node is not a child of this node")
        except KeyError:
            raise ValueError("given child node is not a child of this node")

    def __contains__(self, item):
        return item in self._children

    @property
    def item(self):
        """The item contained in this node."""
        return self._item

    @property
    def count(self):
        """The count associated with this node's item."""
        return self._count

    def increment(self):
        """Increments the count associated with this node's item."""
        if self._count is None:
            raise ValueError("Root nodes have no associated count.")
        self._count += 1

    @property
    def root(self):
        """True if this node is the root of a tree; false if otherwise."""
        return self._item is None and self._count is None

    @property
    def leaf(self):
        """True if this node is a leaf in the tree; false if otherwise."""
        return len(self._children) == 0

    def parent():
        doc = "The node's parent."
        def fget(self):
            return self._parent
        def fset(self, node):
            if node is not None and not isinstance(node, FPNode):
                raise TypeError("A node must have an FPNode as a parent.")
            self._parent = node
        return locals()
    parent = property(**parent())

    def neighbor():
        doc = """
        The node's neighbor; the one with the same value that is "to the right"
        of it in the tree.
        """
        def fget(self):
            return self._neighbor
        def fset(self, node):
            if node is not None and not isinstance(node, FPNode):
                raise TypeError("A node must have an FPNode as a neighbor.")
            self._neighbor = node
        return locals()
    neighbor = property(**neighbor())

    @property
    def children(self):
        """The nodes that are children of this node."""
        return tuple(self._children.itervalues())

    def inspect(self, depth=0):
        print ('  ' * depth) + repr(self)
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return "<%s (root)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)

class FPTree:
    """
    An FP tree.
    This object may only store transaction items that are hashable (i.e., all
    items must be valid as dictionary keys or set members).
    """
    def __init__(self):
        # The root node of the tree.
        self._root = FPNode(None, None)

        # A dictionary mapping items to the head and tail (as pair) of a path of
        # "neighbors" that will hit every node containing that item.
        self._routes = dict()

    @property
    def root(self):
        """The root node of the tree."""
        return self._root

    def add(self, transaction):
        """
        Adds a transaction to the tree.
        """
        point = self._root
        for item in transaction:
            next_point = point.search_child(item)
            if next_point:
                # There is already a node in this tree for the current
                # transaction item; reuse it.
                next_point.increment()
            else:
                # Create a new point and add it as a child of the point we're
                # currently looking at.
                next_point = FPNode(item)
                point.add_child(next_point)

                # Update the route of nodes that contain this item to include
                # our new node.
                self.update_route(next_point)
            point = next_point

    def update_route(self, point):
        """Add the given node to the route through all nodes for its item."""

        route = self._routes.get(point.item, None)
        # existing item
        if route:
            route[1].neighbor = point # route[1] is the tail
            route[1] = point
        # new item
        else:
            self._routes[point.item] = [point, point]

    def items(self):
        """
        Generate one 2-tuples for each item represented in the tree. The first
        element of the tuple is the item itself, and the second element is a
        generator that will yield the nodes in the tree that belong to the item.
        """
        for item in self._routes.iterkeys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        """
        Generates the sequence of nodes that contain the given item.
        """

        try:
            node = self._routes[item][0]
        except KeyError:
            return

        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        """Generates the prefix paths that end with the given item."""

        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print 'Tree:'
        self.root.inspect(1)

        print
        print 'Routes:'
        for item, nodes in self.items():
            print '  %r' % item
            for node in nodes:
                print '    %r' % node

    def remove(self, node):
        """Called when `node` is removed from the tree; performs cleanup."""
        node.parent.remove(node)
        head, tail = self._routes[node.item]
        if node is head:
            if node is tail or not node.neighbor:
                # It was the sole node.
                del self._routes[node.item]
            else:
                self._routes[node.item] = self.Route(node.neighbor, tail)
        else:
            for n in self.nodes(node.item):
                if n.neighbor is node:
                    n.neighbor = node.neighbor # skip over
                    if node is tail:
                        self._routes[node.item] = [head, n]
                    break


class FPGrowth:
    def __init__(self, sample, min_support):
        self.size = len(sample)
        self.minimum_support = self.size * min_support
        logging.info("Building FP tree started ...")
        raw_data_size = sys.getsizeof(sample)
        start_time = time.time()
        self._build_frequent_tree(sample)
        tree_size = sys.getsizeof(self.fp_tree)
        time_elapsed = 1000.0*(time.time() - start_time)
        logging.info('Building FP tree completed. Processed %d samples in %.1f ms (%.2f ms/sample), compressed ratio: %.4f%%',
                     len(sample), time_elapsed, time_elapsed / len(sample), 100.0 * tree_size / raw_data_size)

    def _build_frequent_tree(self, sample):
        support = defaultdict(int) # mapping from items to their supports
        processed_transactions = list()
        # Load the passed-in transactions and count the support that individual item have.
        for transaction in sample:
            processed = list()
            for item in transaction:
                support[item] += 1
                processed.append(item)
            processed_transactions.append(processed)
        self.minimum_support *= len(sample)

        # Remove infrequent items from the item support dictionary.
        self.support = dict((item, support) for item, support in support.iteritems() if support > self.minimum_support)
        # Filter and sort transaction by support
        def clean_transaction(transaction):
            transaction = filter(lambda x: x in self.support, transaction)
            transaction.sort(key=lambda x: self.support[x], reverse=True)
            return transaction

        self.fp_tree = FPTree()
        # Scan input data
        for transaction in imap(clean_transaction, processed_transactions):
            self.fp_tree.add(transaction)

    def condition_tree_from_paths(self, paths):
        condition_tree = FPTree()
        condition_item = None
        pattern = set()

        for path in paths:
            if condition_item is None:
                conition_item = path[-1].item

            point = tree.root
            for node in path:
                next_point = ppint.search(node.item)
                # Add a new node to the tree
                if not next_point:
                    # Add a new node to the tree
                    pattern.add(node.item)
                    count = node.count if node.item == condition_item else 0
                    next_point = FPNode(node.item, count)
                    point.add(next_point)
                    condtion_tree.update_route(next_point)
                point = next_point

        assert condition-item is not None

        # Calculate the counts of the non-leaf nodes
        for path in condition_tree.prefix_paths(conditon_item):
            count = path[-1].count
            for node in reversed(path[:-1]):
                node._count += count

        for item in pattern:
            support = sum(n.count for n in condition_tree.nodes(item))
            if support < self.min_support:
                for node in condition_tree.nodes(item):
                    if node.parent:
                        node.parent.remove_child(node)

        # Finally, remove the nodes corresponding to the item fro which this
        # conditional tree was generated
        for node in condition_tree.nodes(condtion_item):
            if node.parent:
                node.parent.remove_node(node)

        return condition_tree

    def _find_pattern_with_suffix(self, tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= self.min_support and item not in suffix:
                # new qualified candidate
                new_suffix = [item] + suffix
                yield (new_suffix, support)

                condition_tree = self.condition_tree_from_paths(tree.prefix_paths(item))
                for pattern, support in self._find_pattern_with_suffix(condition_tree, new_suffix):
                    yield pattern, support

    def find_frequent_pattern(self):
        for result in self._find_pattern_with_support(self.fp_tree, []):
            yield result

def load_data(fname):
    file = open(fname, 'rU')
    input_data = list()
    for line in file:
        line = line.strip().strip(',')                        # Remove trailing/leading comma
        record = tuple(line.split(','))
        input_data.append(record)
    return input_data

def main():
    optparser = OptionParser()
    optparser.add_option('-f', '--file',
                         dest='input',
                         help='input data filename',
                         default=None)
    optparser.add_option('-s', '--min_support',
                         dest='minS',
                         help='minimum support value',
                         default=0.15,
                         type='float')
    optparser.add_option('-c', '--min_confidence',
                         dest='minC',
                         help='minimum confidence value',
                         default=0.6,
                         type='float')

    (options, args) = optparser.parse_args()
    logging.basicConfig(filename='fp_growth.log',
                    format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)

    input_file = None
    if options.input is None:
        input_file = sys.stdin
    elif options.input is not None:
        input_data = load_data(options.input)
    else:
        print 'No dataset filename specified, system with exit\n'
        sys.exit('System will exit')
    logging.info("FP growth Started ....")
    start_time = time.time()
    min_support = options.minS
    min_confidence = options.minC
    fp_growth = FPGrowth(input_data, min_support)
    frequent_pattern = fp_growth.find_frequent_pattern()
    end_time = time.time()
    time_elapsed = 1000.0 * (end_time - start_time)
    logging.info("FP growth completed in %.1f ms, mined frequent pattern with minimum support %.4f" % (time_elapsed, min_support))

if __name__ == "__main__":
    main()
