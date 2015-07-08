#! /usr/bin/python

from collections import defaultdict
from itertools import imap
from fp_tree_struct import FPTree, FPNode
import logging
import os
import sys
import time
from itertools import chain, combinations

def subsets(arr):
    """
    Returns non empty subsets of arr
    """
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

class FPGrowth:
    def __init__(self, sample, min_support):
        self.minimum_support = min_support
        logging.info("Building FP tree started ...")
        raw_data_size = sys.getsizeof(sample)
        start_time = time.time()
        self._build_frequent_tree(sample)
        tree_size = sys.getsizeof(self.fp_tree)
        time_elapsed = 1000.0*(time.time() - start_time)
        logging.info('Building FP tree completed. Processed %d samples in %.1f ms (%.2f ms/sample), compressed ratio: %.4f%%',
                     len(sample), time_elapsed, time_elapsed / len(sample), 100.0 * tree_size / raw_data_size)

    def _build_frequent_tree(self, sample):
        support = defaultdict(lambda: 0) # mapping from items to their supports
        processed_transactions = list()
        # Load the passed-in transactions and count the support that individual item have.
        for transaction in sample:
            processed = list()
            for item in transaction.strip(',').split(','):
                support[item] += 1
                processed.append(item)
            processed_transactions.append(processed)
        self.minimum_support *= len(sample)

        # Remove infrequent items from the item support dictionary.
        self.freq_items = dict((item, support) for item, support in support.iteritems() if support > self.minimum_support)
        # Build our FP-tree. Before any transactions can be added to the tree, they
        # must be stripped of infrequent items and their surviving items must be
        # sorted in decreasing order of frequency.
        def clean_transaction(transaction):
            transaction = filter(lambda x: x in self.freq_items, transaction)
            transaction.sort(key=lambda x: self.freq_items[x], reverse=True)
            return transaction

        self.fp_tree = FPTree()
        for transaction in imap(clean_transaction, processed_transactions):
            self.fp_tree.add(transaction)

    def _conditional_tree_from_paths(self, paths):
        """
        Builds a conditional FP-tree from the given prefix paths.
        """
        tree = FPTree()
        condition_item = None
        items = set()

        # Import the nodes in the paths into the new tree. Only the counts of the
        # leaf notes matter; the remaining counts will be reconstructed from the
        # leaf counts.
        for path in paths:
            if condition_item is None:
                condition_item = path[-1].item

            point = tree.root
            for node in path:
                next_point = point.search(node.item)
                if not next_point:
                    # Add a new node to the tree.
                    items.add(node.item)
                    count = node.count if node.item == condition_item else 0
                    next_point = FPNode(tree, node.item, count)
                    point.add(next_point)
                    tree.update_route(next_point)
                point = next_point

        assert condition_item is not None

        # Calculate the counts of the non-leaf nodes.
        for path in tree.prefix_paths(condition_item):
            count = path[-1].count
            for node in reversed(path[:-1]):
                node._count += count

        # Eliminate the nodes for any items that are no longer frequent.
        for item in items:
            support = sum(n.count for n in tree.nodes(item))
            if support < self.minimum_support:
                # Doesn't make the cut anymore
                for node in tree.nodes(item):
                    if node.parent is not None:
                        node.parent.remove(node)

        # Finally, remove the nodes corresponding to the item for which this
        # conditional tree was generated.
        for node in tree.nodes(condition_item):
            if node.parent is not None: # the node might already be an orphan
                node.parent.remove(node)

        return tree

    def find_with_suffix(self, tree, suffix):
        """
        Find frequent itemsets in the given transactions using FP-growth. This
        function returns a generator instead of an eagerly-populated list of items.
        The `transactions` parameter can be any iterable of iterables of items.
        `minimum_support` should be an integer specifying the minimum number of
        occurrences of an itemset for it to be accepted.
        Each item must be hashable (i.e., it must be valid as a member of a
        dictionary or a set).
        If `include_support` is true, yield (itemset, support) pairs instead of
        just the itemsets.
        """
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support > self.minimum_support and item not in suffix:
                # New winner!
                found_set = [item] + suffix
                yield (tuple(found_set), support)

                # Build a conditional tree and recursively search for frequent
                # itemsets within it.
                cond_tree = self._conditional_tree_from_paths(tree.prefix_paths(item))
                for s in self.find_with_suffix(cond_tree, found_set):
                    yield s # pass along the good news to our caller

    def run(self):
        # Search for frequent itemsets, and yield the results we find.
        for itemset in self.find_with_suffix(self.fp_tree, []):
            yield itemset

def main():
    from optparse import OptionParser
    p = OptionParser(usage='%prog --help')
    p.add_option('-c', '--minimum-confidence', dest='minconf', type='float',
                 help='Minimum rule confidence (default: 0.85)')
    p.add_option('-s', '--minimum-support', dest='minsup', type='float',
                 help='Minimum itemset support (default: 0.15)')
    p.set_defaults(minsup=2)
    p.add_option('-f', '--file', dest='filename', type='string',
        help='Data filename')
    options, args = p.parse_args()

    if not options.filename :
        p.error('must provide the path to a data file to read')

    f = open(options.filename)
    logging.basicConfig(filename='frequent_pattern.log',
                        format='%(levelname)s: %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)
    try:
        start_time = time.time()
        transactions = f.read().splitlines()
        fp_growth = FPGrowth(transactions, options.minsup)
        logging.info("Mining frequent Pattern started")
        freq_pattern = dict()
        for item, support in fp_growth.run():
            freq_pattern[item] = support
            print "(%s : %d)" % (item, support)
        end_time = time.time()
        logging.info("Found %d frequent pattern with minimum support %f, time elapsed %.1f ms", len(freq_pattern), options.minsup, 1000.0 * (end_time - start_time))
    finally:
        f.close()

if __name__ == '__main__':
    main()
