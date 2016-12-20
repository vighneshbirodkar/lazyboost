from sklearn import datasets
from lazyboost import AdaBoost
import numpy as np
from matplotlib import pyplot as plt


X, y = datasets.load_breast_cancer(return_X_y=True)
index = np.logical_or(y == 0, y == 1)
X, y = X[index], y[index]


clf = AdaBoost(T=100, basetype='weaklinear', transform_classes=True)
clf.fit(X, y)

weaklinear_scores = []
for i in range(100):
    clf.Tlimit = i + 1
    weaklinear_scores.append(clf.score(X, y))

plt.plot(1 - np.array(weaklinear_scores), linewidth=2,
         label='Adaboost using weak linear classifier')

clf = AdaBoost(T=100, basetype='subopt', transform_classes=True)
clf.fit(X, y)

subopt_scores = []
for i in range(100):
    clf.Tlimit = i
    subopt_scores.append(clf.score(X, y))

plt.plot(1 - np.array(subopt_scores), linewidth=2, label='Regular Adaboost')
plt.xlabel('T', fontsize=12)
plt.ylabel('Training Error', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()
