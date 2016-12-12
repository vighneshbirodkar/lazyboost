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
    path = os.path.expanduser('~/data/ionoshphere/ionosphere.data.txt')
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

clf = AdaBoost()
gsv = GridSearchCV(estimator=clf, param_grid={'subopt': [2]}, n_jobs=4)

gsv.fit(X, y)
error = 1 - gsv.cv_results_['mean_test_score'][0]
print('Error = %f' % error)
