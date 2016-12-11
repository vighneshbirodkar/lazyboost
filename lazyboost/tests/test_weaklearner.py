from lazyboost import WeakLinearClassifier
import numpy as np


def test_classify():
    rng = np.random.RandomState(13)
    m = 1000
    X = rng.rand(m, 10)
    r = np.sample(10)
    y = np.zeros_like(1000, dtype=np.int)
    y[r >= (0.5/m)] = 3
    y[r < (0.5/m)] = 7

    clf = WeakLinearClassifier()
    clf.fit(X, y)
