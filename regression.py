#! /usr/bin/python

import sklearn.datasets
import sklearn.linear_model
import numpy.random
import numpy.linalg
import matplotlib.pyplot as plot

def load_data():
    boston = sklearn.datasets.load_boston()
    sample_ratio = 0.5
    size = len(boston.target)
    train_sample_size = int(size * sample_ratio)
    shuffle_idx = range(size)
    numpy.random.shuffle(shuffle_idx)
    train_features = boston.data[shuffle_idx[:train_sample_size]]
    train_targets  = boston.target[shuffle_idx[:train_sample_size]]
    test_features  = boston.data[shuffle_idx[train_sample_size:]]
    test_targets   = boston.target[shuffle_idx[train_sample_size:]]
    return train_features, train_targets, test_features, test_targets

def elastic_net(train_features, train_targets, test_features, test_targets):
    # Train
    elastic_model = sklearn.linear_model.ElasticNetCV(alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 10])
    elastic_model.fit(train_features, train_targets)
    print "Alpha = ", elastic_model.alpha_
    predict_targets = elastic_model.predict(test_features)
    n_test_sample = len(test_targets)
    X = range(n_test_sample)
    # validation
    error = numpy.linalg.norm(predict_targets - test_targets, ord = 1) / n_test_sample
    print "ElasticNet (Boston) Error: %.2f" % (error)
    #Draw
    plot.plot(X, predict_targets, 'r--', label='Predict Price')
    plot.plot(X, test_targets, 'g', label='True Price')
    legend = plot.legend()
    plot.title('ElasticNet (Boston)')
    plot.show()

def ridge(train_features, train_targets, test_features, test_targets):
    # Train
    ridge_model = sklearn.linear_model.RidgeCV(alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 10])
    ridge_model.fit(train_features, train_targets)
    print "Alpha = ", ridge_model.alpha_
    predict_targets = ridge_model.predict(test_features)
    n_test_sample = len(test_targets)
    X = range(n_test_sample)
    # validation
    error = numpy.linalg.norm(predict_targets - test_targets, ord = 1) / n_test_sample
    print "Ridge (Boston) Error: %.2f" % (error)
    #Draw
    plot.plot(X, predict_targets, 'r--', label='Predict Price')
    plot.plot(X, test_targets, 'g', label='True Price')
    legend = plot.legend()
    plot.title('Ridge (Boston)')
    plot.show()

def lasso(train_features, train_targets, test_features, test_targets):
    # Train
    lasso_model    = sklearn.linear_model.LassoCV(alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 10])
    lasso_model.fit(train_features, train_targets)
    print "Alpha = ", lasso_model.alpha_
    # Predict
    predict_targets = lasso_model.predict(test_features)
    n_test_sample = len(test_targets)
    X = range(n_test_sample)
    # validation
    error = numpy.linalg.norm(predict_targets - test_targets, ord = 1) / n_test_sample
    print "Lasso(Boston) Error: %.2f" % (error)
    #Draw
    plot.plot(X, predict_targets, 'r--', label='Predict Price')
    plot.plot(X, test_targets, 'g', label='True Price')
    legend = plot.legend()
    plot.title('Lasso (Boston)')
    plot.show()


if __name__ == '__main__':
    train_features, train_targets, test_features, test_targets = load_data()
    elastic_net(train_features, train_targets, test_features, test_targets)
    lasso(train_features, train_targets, test_features, test_targets)
    ridge(train_features, train_targets, test_features, test_targets)
