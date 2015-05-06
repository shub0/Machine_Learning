#! /usr/bin/python

import sklearn.svm
import matplotlib.pyplot as plot
import matplotlib.colors
from data_loader import SKLearnData

def svc(train_features, train_targets, test_features, test_targets):
    svc = sklearn.svm.SVC()
    nusvc = sklearn.svm.NuSVC()
    linearsvc = sklearn.svm.LinearSVC()

    svc.fit(train_features, train_targets)
    nusvc.fit(train_features, train_targets)
    linearsvc.fit(train_features, train_targets)

    n_test_samples = len(test_targets)

    correct_count_svc = 0
    svc_predict_targets = svc.predict(test_features)
    for index in range(n_test_samples):
        if svc_predict_targets[index] == test_targets[index]:
            correct_count_svc += 1
    accuracy = correct_count_svc * 1.0 / n_test_samples
    print 'SVC accuracy: %.2f' % accuracy

    correct_count_nusvc = 0
    nusvc_predict_targets = nusvc.predict(test_features)
    for index in range(n_test_samples):
        if nusvc_predict_targets[index] == test_targets[index]:
            correct_count_nusvc += 1
    accuracy = correct_count_nusvc * 1.0 / n_test_samples
    print 'NuSVC accuracy: %.2f' % accuracy

    correct_count_linear_svc = 0
    linear_svc_predict_targets = linearsvc.predict(test_features)
    for index in range(n_test_samples):
        if linear_svc_predict_targets[index] == test_targets[index]:
            correct_count_linear_svc += 1
    accuracy = correct_count_linear_svc * 1.0 / n_test_samples
    print 'LinearSVC accuracy: %.2f' % accuracy

def iris_test():
    data_loader = SKLearnData()
    iris_data = data_loader.load_iris()
    print 'Testing SVM classification using Iris data now ...'
    iris_train_features, iris_train_targets, iris_test_features, iris_test_targets = data_loader.random_data(iris_data, 0.5)
    svc(iris_train_features,iris_train_targets, iris_test_features, iris_test_targets)
    print 'Testing Iris data completed'

if __name__ == '__main__':
    iris_test()
