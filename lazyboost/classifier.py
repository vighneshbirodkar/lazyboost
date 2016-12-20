"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .baseclassifier import WeakLinearClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import random


class AdaBoost(BaseEstimator, ClassifierMixin):
    delta_list = []

    def __init__(self, T=200, verbose=False, basetype='subopt', subopt=1,
                 transform_classes=False):
        self.T = T
        self.verbose = verbose
        self.basetype = basetype
        self.subopt = subopt
        self.gamma = np.inf
        self.Tlimit = np.inf
        self.transform_classes = transform_classes

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if self.transform_classes:
            if len(self.classes_) != 2:
                raise NotImplementedError()

            self.class_plus = self.classes_[0]
            self.class_minus = self.classes_[1]

            y_orig = y
            y = np.copy(y)
            y[y_orig == self.class_plus] = 1
            y[y_orig == self.class_minus] = -1

        assert np.all(np.unique(y) == np.array([-1, +1], dtype=np.int))
        D = np.ones_like(y, dtype=np.float)
        D = D/len(D)
        self.Z_list = []
        self.classifiers = []
        self.alphas = []
        self.gammas = []

        delta = np.inf
        for t in range(self.T):
            assert np.isclose(np.sum(D), 1)

            if self.basetype == 'weaklinear':
                h_t = WeakLinearClassifier(verbose=self.verbose).fit(X, y, D)
            elif self.basetype == 'subopt':
                X_fit = X.copy()
                for i in range(self.subopt):
                    h_t = DecisionTreeClassifier(max_depth=1)
                    h_t.fit(X_fit, y, D)
                    selected_feature = np.argmax(h_t.feature_importances_)
                    X_fit[:, selected_feature] = 0.0
            else:
                raise ValueError('unknown bastype')
            predictions = h_t.predict(X)
            epsilon_t = np.sum(D[predictions != y])
            self.gammas.append(np.min([self.gamma,  0.5 - epsilon_t]))

            delta = min(delta, 0.5 - epsilon_t)
            if np.isclose(epsilon_t, 0):
                if t == 0:
                    self.classifiers.append(h_t)
                    self.alphas.append(1)
                break
            #epsilon_t = np.max(epsilon_t, 1e-16)

            if self.basetype == 'weaklinear' and epsilon_t > 0.5:
                raise RuntimeError('Base classifier error = %f' % epsilon_t)
            if self.verbose:
                print('t = %d error = %f' % (t, epsilon_t))

            alpha_t = 0.5 * np.log((1 - epsilon_t)/(epsilon_t))

            Z_t = 2*np.sqrt(epsilon_t*(1 - epsilon_t))
            self.Z_list.append(Z_t)
            self.classifiers.append(h_t)
            self.alphas.append(alpha_t)

            predictions = h_t.predict(X)
            D = D * (np.exp(-alpha_t*y*predictions))
            D = D/Z_t

        return self

    def predict(self, X):
        check_is_fitted(self, ['classifiers', 'alphas'])

        predictions = np.zeros(X.shape[0], dtype=np.float)
        for i in range(min(len(self.classifiers), self.Tlimit)):
            predictions += (self.alphas[i]*self.classifiers[i].predict(X))

        ypred = np.sign(predictions)

        y = np.copy(ypred)
        y[ypred == 1] = self.class_plus
        y[ypred == -1] = self.class_minus

        self._last_predictions = predictions
        return y

    def get_margin(self, X, y):
        check_is_fitted(self, ['classifiers', 'alphas'])

        predictions = np.zeros(X.shape[0], dtype=np.float)
        for i in range(min(len(self.classifiers), self.Tlimit)):
            predictions += (self.alphas[i]*self.classifiers[i].predict(X))

        y_orig = y
        y = np.copy(y_orig)
        y[y_orig == self.class_plus] = 1
        y[y_orig == self.class_minus] = -1

        return (predictions*y)/(np.sum(np.abs(self.alphas)))


class AdaBoostCV(BaseEstimator):
    def __init__(self, adaboost, num_splits=4, T=1000):
        self.adaboost = clone(adaboost)
        self.num_splits = num_splits
        self.T = T

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise NotImplementedError()

        self.class_plus = self.classes_[0]
        self.class_minus = self.classes_[1]

        y_orig = y
        y = np.copy(y)
        y[y_orig == self.class_plus] = 1
        y[y_orig == self.class_minus] = -1

        kf = KFold(n_splits=self.num_splits, random_state=1)
        self.adaboost.T = self.T

        self._test_scores = np.zeros(self.T)
        self._gammas = np.zeros(self.T)
        for train_index, test_index in kf.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            self.adaboost.fit(X_train, y_train)

            predictions = np.zeros_like(y_test, dtype=np.float)
            for t in range(0, len(self.adaboost.classifiers)):
                alpha = self.adaboost.alphas[t]
                pred = self.adaboost.classifiers[t].predict(X_test)
                predictions += alpha*pred

                #print(np.sign(predictions)[:10], y_test[:10], pred[:10])
                num_correct = np.sum(np.sign(predictions) == y_test)
                self._test_scores[t] += float(num_correct)/y_test.shape[0]

                self._gammas[t] += min(self.adaboost.gammas[:t+1])

        self._test_scores /= self.num_splits
        self._gammas /= self.num_splits

    def predict(self, X):
        raise NotImplementedError()
