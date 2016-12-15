from lazyboost import AdaBoost
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
import os
import csv

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


def evaluate_adaboost(Trange, X, y):
    errors = []
    for T in Trange:
        clf = AdaBoost(T=T)
        gsv = GridSearchCV(estimator=clf, cv=10, param_grid={'subopt': [i+1]},
                           n_jobs=4)
        gsv.fit(X, y)
        error = 1 - gsv.cv_results_['mean_test_score'][0]
        errors.append(error)
        print('T = %d error = %f' % (T, error))
    return errors

for i in range(5):
    print('Subopt %d' % (i + 1))
    Trange = [100, 200, 500, 1000]
    errors = evaluate_adaboost(Trange, X, y)
    plt.plot(Trange, errors, label='subopt = %d' % (i + 1))

plt.grid(True)
plt.legend()
plt.show()
