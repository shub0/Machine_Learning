#! /usr/bin/python

from data_loader import SKLearnData
import regression
import sklearn.linear_model

def load_data():
    data_loader = SKLearnData()
    iris_data = data_loader.load_iris()
    size = len(iris_data.target)
    def filter_func(index):
        if iris_data.target[index] == 1:
            return False
        return True
    index = filter(filter_func, range(size))
    iris_data.data = iris_data.data[index]
    iris_data.target = iris_data.target[index] / 2
    iris_train_features, iris_train_targets, iris_test_features, iris_test_targets = data_loader.random_data(iris_data, 0.5)
    return iris_train_features, iris_train_targets, iris_test_features, iris_test_targets

def lasso_classification(train_features, train_targets, test_features, test_targets):
    lasso_model = sklearn.linear_model.LassoCV(alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 10])
    lasso_model.fit(train_features, train_targets)
    print 'Alpha = ', lasso_model.alpha_
    def classify(target):
        if target > 0.5:
            return 1
        return 0
    predict_targets = map(classify, lasso_model.predict(test_features))
    n_test_sample = len(test_targets)
    correct_count = 0
    for index in range(n_test_sample):
        if predict_targets[index] == test_targets[index]:
            correct_count += 1
    print 'Accuracy: %.2f' % ( correct_count * 1.0 / n_test_sample )

def iris_test():
    iris_train_features, iris_train_targets, iris_test_features, iris_test_targets = load_data()
    lasso_classification(iris_train_features, iris_train_targets, iris_test_features, iris_test_targets)

if __name__ == '__main__':
    iris_test()
