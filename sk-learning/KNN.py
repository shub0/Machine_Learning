#! /usr/bin/python

import sklearn.neighbors
import matplotlib.pyplot as plot
import matplotlib.colors as color
from data_loader import SKLearnData

def KNN(train_features, train_targets, test_features, test_targets):
    n_neighbors = 5
    n_test_samples = len(test_targets)
    for weight in ['uniform', 'distance']:
        knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
        knn_model.fit(train_features, train_targets)
        predict_targets = knn_model.predict(test_features)
        correct_count = 0
        for index in range(n_test_samples):
            if predict_targets[index] == test_targets[index]:
                correct_count += 1
        accuracy = correct_count * 1.0 / n_test_samples
        print 'KNN Accuracy [weight = %s]: %.2f' % (weight, accuracy)
        cmap_bold = color.ListedColormap(['red', 'blue', 'green'])
        X_test = test_features[:, 2:4]
        X_train = train_features[:, 2:4]
        plot.scatter(X_train[:, 0], X_train[:, 1], label = 'train samples', marker='o', c = train_targets, cmap=cmap_bold,)
        plot.scatter(X_test[:,0], X_test[:, 1], label = 'test samples', marker='+', c = predict_targets, cmap=cmap_bold)
        legend = plot.legend()
        plot.title("K Neighbors Classifier (Iris) [weight = %s]" %(weight))
        plot.savefig("K Neighbors Classifier (Iris) [weight = %s].png" %(weight), format='png')
        plot.show()

def test_iris():
    data_loader = SKLearnData()
    iris_data = data_loader.load_iris()
    iris_train_features, iris_train_targets, iris_test_features, iris_test_targets = data_loader.random_data(iris_data, 0.8)
    KNN(iris_train_features, iris_train_targets, iris_test_features, iris_test_targets)

if __name__ == '__main__':
    test_iris()
