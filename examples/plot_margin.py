from lazyboost import AdaBoost, AdaBoostCV
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
import os
import csv
from skimage import io

datatype = 'breastcancer'

data = datasets.load_breast_cancer()
X = data.data
y = data.target

clf = AdaBoost(T=100, transform_classes=True, subopt=6)
clf.fit(X, y)
clf.predict(X)
margin = clf.get_margin(X, y)
plt.plot(np.sort(margin))
print(clf.gammas[-1])

plt.show()
