#! /usr/bin/python

import sklearn.datasets
import sklearn.linear_model
import numpy.random
import numpy.linalg
import matplotlib.pyplot as plot

def load_data(data_sets, train_sample_ratio):
    size = len(data_sets.target)
    train_sample_size = int(size * train_sample_ratio)
    shuffle_idx = range(size)
    numpy.random.shuffle(shuffle_idx)
    train_features = data_sets.data[shuffle_idx[:train_sample_size]]
    train_targets  = data_sets.target[shuffle_idx[:train_sample_size]]
    test_features  = data_sets.data[shuffle_idx[train_sample_size:]]
    test_targets   = data_sets.target[shuffle_idx[train_sample_size:]]
    return train_features, train_targets, test_features, test_targets


def linear_regression(train_features, train_targets, test_features, test_targets):
    # Train
    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(train_features, train_targets)
    predict_targets = linear_model.predict(test_features)
    n_test_sample = len(test_targets)
    X = range(n_test_sample)
    # validation
    error = numpy.linalg.norm(predict_targets - test_targets, ord = 1) / n_test_sample
    print "Linear Regression Error: %.2f" % (error)
    #Draw
    plot.plot(X, predict_targets, 'r--', label='Predict Price')
    plot.plot(X, test_targets, 'g', label='True Price')
    legend = plot.legend()
    plot.title('Linear Regression')
    plot.show()

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
    print "ElasticNet  Error: %.2f" % (error)
    #Draw
    plot.plot(X, predict_targets, 'r--', label='Predict Price')
    plot.plot(X, test_targets, 'g', label='True Price')
    legend = plot.legend()
    plot.title('ElasticNet')
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
    print "Ridge Error: %.2f" % (error)
    #Draw
    plot.plot(X, predict_targets, 'r--', label='Predict Price')
    plot.plot(X, test_targets, 'g', label='True Price')
    legend = plot.legend()
    plot.title('Ridge')
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
    print "Lasso Error: %.2f" % (error)
    #Draw
    plot.plot(X, predict_targets, 'r--', label='Predict Price')
    plot.plot(X, test_targets, 'g', label='True Price')
    legend = plot.legend()
    plot.title('Lasso')
    plot.show()

def logistic_regression(train_features, train_targets, test_features, test_targets):
    logistic_model = sklearn.linear_model.LogisticRegression()
    logistic_model.fit(train_features, train_targets)
    predict_targets = logistic_model.predict(test_features)
    n_test_samples = len(test_targets)
    correct_count = 0
    for index in range(n_test_samples):
        if predict_targets[index] == test_targets[index]:
            correct_count += 1
    accuracy = correct_count * 1.0 / n_test_samples
    print "Logistic Regression Accuracy: %.2f" % accuracy

    X = range(n_test_samples)
    plot.subplot(211)
    plot.title('Logistic Regression')
    plot.plot(X, predict_targets, 'ro-', label = 'Predict Labels')
    plot.ylabel('Predict Class')
    legend = plot.legend()

    plot.subplot(212)
    plot.plot(X, test_targets, 'go-', label = 'True Labels')
    plot.ylabel('True Class')
    legend = plot.legend()
    plot.show()

def boston_test():
    boston_data = sklearn.datasets.load_boston()
    boston_train_features, boston_train_targets, boston_test_features, boston_test_targets = load_data(boston_data, 0.5)
    print 'Testing regression using Boston data now ...'
    linear_regression(boston_train_features, boston_train_targets, boston_test_features, boston_test_targets)
    elastic_net(boston_train_features, boston_train_targets, boston_test_features, boston_test_targets)
    lasso(boston_train_features, boston_train_targets, boston_test_features, boston_test_targets)
    ridge(boston_train_features, boston_train_targets, boston_test_features, boston_test_targets)
    print 'Polynomina Linear Regression (Degree = 2)'
    polynomina_data = sklearn.preprocessing.PolynomialFeatures(degree=2).fit_transform(boston_data.data)
    polynomina_boston_train_features, polynomina_boston_train_targets, polynomina_boston_test_features, polynomina_boston_test_targets = load_data(boston_data, 0.5)
    linear_regression(polynomina_boston_train_features, boston_train_targets, polynomina_boston_test_features, boston_test_targets)
    print 'Testing Boston data completed'

def iris_test():
    iris_data = sklearn.datasets.load_iris()
    print 'Testing regression using Iris data now ...'
    iris_train_features, iris_train_targets, iris_test_features, iris_test_targets = load_data(iris_data, 0.5)
    logistic_regression(iris_train_features,iris_train_targets, iris_test_features, iris_test_targets)
    print 'Testing Iris data completed'

if __name__ == '__main__':
    boston_test()
    #iris_test()
