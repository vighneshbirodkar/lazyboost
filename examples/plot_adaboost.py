from lazyboost import AdaBoost, AdaBoostCV
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
import os
import csv
from skimage import io
import random

datatype = 'digits'

if datatype == 'breastcancer':
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

elif datatype == 'ionosphere':
    path = os.path.expanduser('~/data/ionosphere/ionosphere.data.txt')
    with open(path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    data = [row[:-1] for row in rows]
    labels = [row[-1] for row in rows]
    labels = [1 if l == 'g' else 0 for l in labels]

    X = np.array(data)
    y = np.array(labels)

elif datatype == 'digits':
    X, y = datasets.load_digits(n_class=2, return_X_y=True)

elif datatype == 'ocr':
    path = os.path.expanduser('~/data/ocr/optdigits.tra')
    with open(path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    data = [row[:-1] for row in rows]
    print(len(data))
    labels = [int(row[-1]) for row in rows]

    X = np.array(data)
    y = np.array(labels)
    index = np.logical_or(y == 4, y == 7)

    X, y = X[index, :], y[index]

elif datatype == 'iris':
    X, y = datasets.load_iris(return_X_y=True)
    index = np.logical_or(y == 0, y == 1)
    X, y = X[index], y[index]

elif datatype == 'mnist':
    path = os.path.expanduser('~/data/mnist_png/training/')
    X, y = [], []
    for img in os.listdir(path + '4'):
        X.append(io.imread(path + '/4/' + img).flatten())
        y.append(4)
    for img in os.listdir(path + '7'):
        X.append(io.imread(path + '/7/' + img).flatten())
        y.append(7)

    X, y = np.array(X), np.array(y)


def evaluate_adaboost_at_range(Trange, subopt, X, y):
    clf = AdaBoost(T=Trange, basetype='subopt', subopt=subopt)
    abcv = AdaBoostCV(clf, num_splits=10)
    abcv.fit(X, y)

    return 1 - abcv._test_scores, abcv._gammas

print(X.shape)


error_fig, error_ax = plt.subplots()
gamma_fig, gamma_ax = plt.subplots()


for i in range(3):

    print('Subopt %d' % (i + 1))
    errors, gammas = evaluate_adaboost_at_range(1000, i + 1, X, y)

    gammas = np.maximum(np.array(gammas), 1e-6)
    error_ax.plot(errors, label='subopt = %d' % (i + 1))
    gamma_ax.plot(np.log(gammas), label='subopt = %d' % (i + 1))


gamma_ax.grid(True)
gamma_ax.legend()
gamma_ax.set_ylabel('$\log(\gamma)$', fontsize=14)
gamma_ax.set_xlabel('T', fontsize=14)

error_ax.grid(True)
error_ax.legend()
error_ax.set_ylabel('CV-Error', fontsize=14)
error_ax.set_xlabel('T', fontsize=14)


plt.show()

