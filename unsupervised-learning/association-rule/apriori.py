"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATAFILE -s min_support  -c min_confidence

    $python apriori.py -f DATAFILE -s 0.15 -c 0.6
"""

import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
import logging
import time

class Apriori:
    def __init__(self, data_iterator, min_support, min_confidence):
        self.transaction_list = list()
        self.item_set = set()
        self.freq_set = defaultdict(int)
        self.min_support = min_support
        self.min_confidence = min_confidence
        for record in data_iterator:
            transaction = frozenset(record)
            self.transaction_list.append(transaction)
            for item in transaction:
                self.item_set.add(frozenset([item]))              # Generate 1-item sets
        self.size = len(self.transaction_list)

    @staticmethod
    def subsets(arr):
        """
        Returns non empty subsets of arr
        """
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

    @staticmethod
    def expand_set(item_set, length):
        """
        Join a set with itself and returns the n-element itemsets
        """
        return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])

    def _filter_item_sets(self, item_set):
        """
        calculates the support for items in the item_set and returns a subset
        of the item_set each of whose elements satisfies the minimum support
        """
        item_set_with_min_support = set()
        local_set = defaultdict(int)

        for item in item_set:
            for transaction in self.transaction_list:
                if item.issubset(transaction):
                    self.freq_set[item] += 1
                    local_set[item] += 1

        for item, count in local_set.items():
            support = float(count) / self.size
            if support >= self.min_support:
                item_set_with_min_support.add(item)

        return item_set_with_min_support

    def _get_support(self, item):
        """
        local function which Returns the support of an item
        """
        return float(self.freq_set[item]) / self.size

    def run(self):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
         - items (tuple, support)
         - rules ((pretuple, posttuple), confidence)
        """

        # Global dictionary which stores (key=size of item_sets, value=n-item_sets) which satisfy min_support
        self.large_set = dict()

        # 1-item sets
        current_set_size_k = self._filter_item_sets(self.item_set)
        k = 1
        while(current_set_size_k != set([])):
            self.large_set[k] = current_set_size_k
            # Continue searching sets with larger size
            k = k + 1
            # Generate new sets with size = K from set with size = k-1
            current_set_size_k = Apriori.expand_set(current_set_size_k, k)
            # Filter sets with min_support
            current_set_size_k = self._filter_item_sets(current_set_size_k)

        self.freq_pattern = list()
        for key, value in self.large_set.items():
            self.freq_pattern.extend([(tuple(item), self._get_support(item)) for item in value])

    def generate_rules(self):
        self.rules = list()
        # ignore 0-item_sets
        for key, value in self.large_set.items()[1:]:
            for item in value:
                _subsets = map(frozenset, [x for x in Apriori.subsets(item)])
                for antecedent in _subsets:
                    consequent = item.difference(antecedent)
                    if len(consequent) > 0:
                        confidence = self._get_support(item) / self._get_support(antecedent)
                        lift = confidence / self._get_support(consequent)
                        if confidence >= self.min_confidence:
                            self.rules.append(((tuple(antecedent), tuple(consequent)), confidence, lift))

    def output(self):
        """
        prints the generated itemsets sorted by support and the confidence rules sorted by confidence
        """
        for item, support in sorted(self.freq_pattern, key=lambda (item, support): support, reverse=True):
            print "item: %s , %.3f" % (str(item), support)
        print "\n------------------------ RULES:"
        """
        for rule, confidence, lift in sorted(self.rules, key=lambda (rule, confidence, lift): confidence):
            pre, post = rule
            print "Rule: %s ==> %s , %.3f, %.3f" % (str(pre), str(post), confidence, lift)
        """
def load_data(fname):
    """
    Function which reads from the file and yields a generator
    """
    file_iter = open(fname, 'rU')
    for line in file_iter:
        line = line.strip().strip(',')                        # Remove trailing/leading comma
        record = frozenset(line.split(','))
        yield record

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
    logging.basicConfig(filename='apriori.log',
                    format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)

    start_time = time.time()
    input_file = None
    if options.input is None:
        input_file = sys.stdin
    elif options.input is not None:
        input_file = load_data(options.input)
    else:
        print 'No dataset filename specified, system with exit\n'
        sys.exit('System will exit')
    logging.info("Apriori Started ....")
    min_support = options.minS
    min_confidence = options.minC
    apriori = Apriori(input_file, min_support, min_confidence)
    apriori.run()
    end_time = time.time()
    apriori.generate_rules()
    logging.info("Apriori completed, %d pattern with minimum support %f, %d rules with minimum confidecne %f, time elapased %.1f ms." % (len(apriori.freq_pattern), min_support, len(apriori.rules), min_confidence, 1000.0 * (end_time - start_time)))
    apriori.output()

if __name__ == "__main__":
    main()
