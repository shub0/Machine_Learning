#! /usr/bin/python

from collections import defaultdict
from itertools import imap
from fp_tree_struct import FPTree, FPNode
import logging
import os
import time

def conditional_tree_from_paths(paths, minimum_support):
    """Builds a conditional FP-tree from the given prefix paths."""
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
                tree._update_route(next_point)
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
        if support < minimum_support:
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


def build_frequent_tree(transactions, minimum_support):
    logging.info("Building FP tree started ...")
    items = defaultdict(lambda: 0) # mapping from items to their supports
    processed_transactions = []
    start_time = time.time()
    # Load the passed-in transactions and count the support that individual
    # items have.
    sample = transactions.splitlines()
    for transaction in sample:
        processed = []
        for item in transaction.strip(',').split(','):
            items[item] += 1
            processed.append(item)
        processed_transactions.append(processed)

    # Remove infrequent items from the item support dictionary.
    items = dict((item, support) for item, support in items.iteritems() if support >= minimum_support)

    # Build our FP-tree. Before any transactions can be added to the tree, they
    # must be stripped of infrequent items and their surviving items must be
    # sorted in decreasing order of frequency.
    def clean_transaction(transaction):
        transaction = filter(lambda x: x in items, transaction)
        transaction.sort(key=lambda x: items[x], reverse=True)
        return transaction

    fp_tree = FPTree()
    for transaction in imap(clean_transaction, processed_transactions):
        fp_tree.add(transaction)
    time_elapsed = 1000.0*(time.time() - start_time)
    logging.info('Building FP tree completed. Processed %d samples in %.1f ms (%.2f ms/sample)', len(sample), time_elapsed, time_elapsed / len(sample))
    return fp_tree

def find_with_suffix(tree, suffix, minimum_support, include_support=False):
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
        if support >= minimum_support and item not in suffix:
            # New winner!
            found_set = [item] + suffix
            yield (found_set, support) if include_support else found_set

            # Build a conditional tree and recursively search for frequent
            # itemsets within it.
            cond_tree = conditional_tree_from_paths(tree.prefix_paths(item), minimum_support)
            for s in find_with_suffix(cond_tree, found_set, minimum_support, include_support):
                yield s # pass along the good news to our caller


def find_frequent_itemsets(master_tree, minimum_support, include_support=False):
    # Search for frequent itemsets, and yield the results we find.
    for itemset in find_with_suffix(master_tree, [], minimum_support, include_support):
        yield itemset

def main():
    from optparse import OptionParser
    p = OptionParser(usage='%prog --help')
    p.add_option('-s', '--minimum-support', dest='minsup', type='int',
        help='Minimum itemset support (default: 2)')
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
        minimum_support = options.minsup
        tree = build_frequent_tree(f.read(), minimum_support)
        """
        for itemset, support in find_frequent_itemsets(tree, minimum_support, True):
            print "{" + ", ".join(itemset) + "} " + str(support)
        """
        suffix = ["17108", "20203"]
        logging.info("Frequent Pattern with Suffix: %s", suffix)
        frequent_pattern = list()
        for itemset, support in find_with_suffix(tree, suffix, minimum_support, True):
            frequent_pattern.append((', '.join(itemset), support))
        frequent_pattern.sort(key=lambda x: x[1], reverse=True)
        logging.info("Found %d frequent pattern with minimum support %d", len(frequent_pattern), minimum_support)
        print '\n'.join([ str(pattern) for pattern in frequent_pattern ])
    finally:
        f.close()

if __name__ == '__main__':
    main()
