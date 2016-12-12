"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .baseclassifier import WeakLinearClassifier
from sklearn.tree import DecisionTreeClassifier


class AdaBoost(BaseEstimator, ClassifierMixin):

    def __init__(self, T=200, verbose=False, basetype='subopt', subopt=1):
        self.T = T
        self.verbose = verbose
        self.basetype = basetype
        self.subopt = subopt

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

        D = np.ones_like(y, dtype=np.float)
        D = D/len(D)
        self.Z_list = []
        self.classifiers = []
        self.alphas = []

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
            #D = D/np.sum(D)
            #print('Z produt = ', np.prod(self.Z_list))

        return self

    def predict(self, X):
        check_is_fitted(self, ['classifiers', 'alphas'])

        predictions = np.zeros(X.shape[0], dtype=np.float)
        for i in range(len(self.classifiers)):
            predictions += (self.alphas[i]*self.classifiers[i].predict(X))

        ypred = np.sign(predictions)

        y = np.copy(ypred)
        y[ypred == 1] = self.class_plus
        y[ypred == -1] = self.class_minus

        return y
