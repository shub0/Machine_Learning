"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATASET.csv -s min_support  -c min_confidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


def subsets(arr):
    """
    Returns non empty subsets of arr
    """
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def items_with_min_support(item_set, transaction_list, min_support, freq_set):
    """
    calculates the support for items in the item_set and returns a subset
    of the item_set each of whose elements satisfies the minimum support
    """
    _item_set = set()
    local_set = defaultdict(int)

    for item in item_set:
        for transaction in transaction_list:
            if item.issubset(transaction):
                freq_set[item] += 1
                local_set[item] += 1

    for item, count in local_set.items():
        support = float(count)/len(transaction_list)
        if support >= min_support:
            _item_set.add(item)

    return _item_set

def join_set(item_set, length):
    """
    Join a set with itself and returns the n-element itemsets
    """
    return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])

def get_item_set_transaction_list(data_iterator):
    transaction_list = list()
    item_set = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transaction_list.append(transaction)
        for item in transaction:
            item_set.add(frozenset([item]))              # Generate 1-item sets
    return item_set, transaction_list

def run_aprirori(data_iter, min_support, min_confidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    item_set, transaction_list = get_item_set_transaction_list(data_iter)

    # Global dictionary which stores (key=n-item_sets,value=support)
    # which satisfy min_support
    freq_set = defaultdict(int)
    large_set = dict()

    # Dictionary which stores Association Rules
    associate_rules = dict()

    current_set_size = items_with_min_support(item_set, transaction_list, min_support, freq_set)
    k = 1
    while(current_set_size != set([])):
        large_set[k] = current_set_size
        # Continue searching sets with larger size
        k = k + 1
        # Generate new sets with size = K
        current_set_size = join_set(current_set_size, k)
        # Filter sets with min_support
        current_set_size = items_with_min_support(current_set_size, transaction_list, min_support, freq_set)

    def get_support(item):
        """
        local function which Returns the support of an item
        """
        return float(freq_set[item])/len(transaction_list)

    output_items = list()
    for key, value in large_set.items():
        output_items.extend([(tuple(item), get_support(item)) for item in value])

    output_rules = []
    for key, value in large_set.items()[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = get_support(item)/get_support(element)
                    lift = confidence / get_support(remain)
                    if confidence >= min_confidence:
                        output_rules.append(((tuple(element), tuple(remain)), confidence, lift))
    return output_items, output_rules


def print_output(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    for item, support in sorted(items, key=lambda (item, support): support):
        print "item: %s , %.3f" % (str(item), support)
    print "\n------------------------ RULES:"
    for rule, confidence, lift in sorted(rules, key=lambda (rule, confidence, lift): confidence):
        pre, post = rule
        print "Rule: %s ==> %s , %.3f, %.3f" % (str(pre), str(post), confidence, lift)

def load_data(fname):
    """Function which reads from the file and yields a generator"""
    file_iter = open(fname, 'rU')
    for line in file_iter:
        line = line.strip().strip(',')                        # Remove trailing/leading comma
        record = frozenset(line.split(','))
        yield record

def main():

    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
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

    input_file = None
    if options.input is None:
            input_file = sys.stdin
    elif options.input is not None:
            input_file = load_data(options.input)
    else:
            print 'No dataset filename specified, system with exit\n'
            sys.exit('System will exit')

    min_support = options.minS
    min_confidence = options.minC

    items, rules = run_aprirori(input_file, min_support, min_confidence)

    print_output(items, rules)

if __name__ == "__main__":
    main()
