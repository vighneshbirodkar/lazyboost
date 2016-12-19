from lazyboost import AdaBoost, AdaBoostCV
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
import os
import csv
import multiprocessing as mp
from functools import partial


datatype = 'ionosphere'

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
    labels = [int(row[-1]) for row in rows]

    X = np.array(data)
    y = np.array(labels)
    index = np.logical_or(y == 4, y == 7)

    X, y = X[index, :], y[index]


def evaluate_adaboost_at_T(T, subopt, X, y):
    clf = AdaBoost(T=T, basetype='subopt', subopt=subopt)
    clf.fit(X, y)
    abcv = AdaBoostCV(clf, num_splits=10)
    abcv.fit(X, y)
    avg_error = 1 - np.mean(abcv._test_scores)
    avg_gamma = np.mean(abcv._gammas)
    print('T = %d avg. error = %f avg. gamma = %f' % (T, avg_error, avg_gamma))

    return avg_error, avg_gamma


def evaluate_adaboost_at_range(Trange, subopt, X, y):
    pool = mp.Pool()
    func = partial(evaluate_adaboost_at_T, X=X, y=y, subopt=subopt)
    errors_and_gammas = pool.map(func, Trange)
    return errors_and_gammas

error_fig, error_ax = plt.subplots()
gamma_fig, gamma_ax = plt.subplots()

for i in range(3):
    print('Subopt %d' % (i + 1))
    Trange = [100, 200, 500, 1000]
    errors_and_gammas = evaluate_adaboost_at_range(Trange, i + 1, X, y)
    errors_and_gammas = np.array(errors_and_gammas)
    errors = errors_and_gammas[:, 0]
    gammas = errors_and_gammas[:, 1]
    error_ax.plot(Trange, errors, label='subopt = %d' % (i + 1))
    gamma_ax.plot(Trange, gammas, label='subopt = %d' % (i + 1))
    

gamma_ax.grid(True)
gamma_ax.legend()
gamma_ax.set_title('Gamma')

error_ax.grid(True)
error_ax.legend()
error_ax.set_title('Error')

plt.show()
