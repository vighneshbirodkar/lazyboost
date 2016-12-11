from sklearn.ensemble import AdaBoostClassifier
from lazyboost import WeakLinearClassifier, AdaBoost
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np


X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=0)


clf = AdaBoost(verbose=True, T=100, basetype='subopt', subopt=1)
clf.fit(X, y)

plt.plot(np.cumprod(clf.Z_list))


clf = AdaBoost(verbose=True, T=100, basetype='subopt', subopt=2)
clf.fit(X, y)

plt.plot(np.cumprod(clf.Z_list))
plt.show()
