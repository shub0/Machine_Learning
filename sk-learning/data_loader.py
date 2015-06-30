#! /usr/bin/python

import sklearn.datasets
import numpy.random

class SKLearnData:
    def load_boston(self):
        return sklearn.datasets.load_boston()

    def load_iris(self):
        return sklearn.datasets.load_iris()

    def random_data(self, data_sets, train_sample_ratio):
        size = len(data_sets.target)
        train_sample_size = int(size * train_sample_ratio)
        shuffle_idx = range(size)
        numpy.random.shuffle(shuffle_idx)
        train_features = data_sets.data[shuffle_idx[:train_sample_size]]
        train_targets  = data_sets.target[shuffle_idx[:train_sample_size]]
        test_features  = data_sets.data[shuffle_idx[train_sample_size:]]
        test_targets   = data_sets.target[shuffle_idx[train_sample_size:]]
        return train_features, train_targets, test_features, test_targets
