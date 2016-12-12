from __future__ import print_function
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


def sample_u(X, rng):
    found = False
    while not found:
        u = rng.rand(X.shape[1])
        projection = np.dot(X, u)

        # All projections are unique
        found = np.all(np.unique(projection).shape == projection.shape)

    # Check adjacent elemements in sorted order to find least difference
    projection_sorted = np.sort(projection)
    projection_shifted = projection_sorted[1:]
    projection_sorted = np.sort(projection_sorted)[:-1]
    diff = projection_shifted - projection_sorted
    e = np.min(diff)
    assert e > 0

    return u, e


class SubOptimalStump(BaseEstimator, ClassifierMixin):
    def __init__(self, rank=1, verbose=False):
        self.rank = rank
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if len(self.classes_) > 2:
            raise NotImplementedError('Only binary classification supported')

        m = len(y)
        p = sample_weight
        if p is None:
            p = np.ones_like(y, dtype=np.float)/m

        assert np.isclose(np.sum(p), 1)
        self.class_plus = self.classes_[0]
        self.class_minus = self.classes_[1]

        y_orig = y
        y = np.copy(y)
        y[y_orig == self.class_plus] = 1
        y[y_orig == self.class_minus] = -1

        n_features = X.shape[1]
        print(y)
        for feature_idx in range(n_features):
            pass


class WeakLinearClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, verbose=False, rseed=None):
        self.verbose = verbose
        self.rng = np.random.RandomState(rseed)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if len(self.classes_) > 2:
            raise NotImplementedError('Only binary classification supported')

        m = len(y)
        p = sample_weight
        if p is None:
            p = np.ones_like(y, dtype=np.float)/m

        assert np.isclose(np.sum(p), 1)
        self.class_plus = self.classes_[0]
        self.class_minus = self.classes_[1]

        y_orig = y
        y = np.copy(y)
        y[y_orig == self.class_plus] = 1
        y[y_orig == self.class_minus] = -1

        if np.abs(np.sum(p * y)) >= 1.0/(2*m):
            if self.verbose:
                print('Trivial Classification case')
            if np.sum(p[y == 1]) >= (np.sum(p[y == -1]) + 1.0/(2*m)):
                self.a = np.zeros(X.shape[1])
                self.b = 1

                assert self._score_sign(X, y, p) > 0.5
                return self
            elif np.sum(p[y == -1]) >= (np.sum(p[y == 1]) + 1.0/(2*m)):
                self.a = np.zeros(X.shape[1])
                self.b = -1

                assert self._score_sign(X, y, p) > 0.5
                return self
            else:
                raise ValueError('Something is wrong')
        else:
            if self.verbose:
                print('Sampling u')
            u, e = sample_u(X, self.rng)
            if self.verbose:
                print('Finished sampling u')
            i = np.where(p > (1.0/m))[0][0]
            epsilon = e/2.0
            a = u
            b = -np.dot(u, X[i]) + epsilon*y[i]

            ypred = np.sign(np.dot(X, a) + b)
            error = float(np.sum(ypred * y * p))
            if np.abs(error) >= 1.0/(2*m):
                if error > 0:
                    self.a = a
                    self.b = b
                else:
                    self.a = -a
                    self.b = -b
            else:
                ypred = np.sign(np.dot(X, a) + b)
                error = float(np.sum(ypred * y * p))

                if error > 0:
                    self.a = a
                    self.b = b
                else:
                    self.a = -a
                    self.b = -b
                self.a = -u
                self.b = np.dot(u, X[i]) + epsilon*y[i]

            assert self._score_sign(X, y, p) > 0.5

        return self

    def _predict_sign(self, X):
        ypred = np.sign(np.dot(X, self.a) + self.b)
        return ypred

    def _score_sign(self, X, y, sample_weights):
        ypred = self._predict_sign(X)
        return accuracy_score(y, ypred, sample_weight=sample_weights)

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['a', 'b'])

        # Input validation
        X = check_array(X)

        ypred = self._predict_sign(X)
        y = np.copy(ypred)
        y[ypred == 1] = self.class_plus
        y[ypred == -1] = self.class_minus

        return y

if __name__ == '__main__':

    clf = DecisionTreeClassifier(max_depth=1)
    X, y = datasets.make_hastie_10_2()
    clf.fit(X, y)
    print(clf.feature_importances_)
