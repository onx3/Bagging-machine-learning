from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import numpy as np



class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, random_state=None):
        """
        Parameters
        ----------
        base_estimator : object or None, optional (default=None)    The base estimator to fit on random subsets of the dataset. 
                                                                    If None, then the base estimator is a decision tree.
        n_estimators : int, optional (default=10)                   The number of base estimators in the ensemble.
        random_state : int or None, optional (default=None)         Controls the randomness of the estimator. 
        """

        #TODO: ...
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.estimators_weights_ = []




    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        rng = np.random.default_rng(self.random_state)

        for i in range(self.n_estimators):
            indices = rng.choice(X.shape[0], X.shape[0], replace=True)

            estimator = self.base_estimator.__class__()
            estimator.fit(X[indices], y[indices])

            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        predictions = np.array([estimator.predict(X) for estimator in self.estimators_]).T

        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)



    def predict_proba(self, X):
        check_is_fitted(self)

        X = check_array(X)

        probabilities = np.array([estimator.predict_proba(X) for estimator in self.estimators_])

        return np.mean(probabilities, axis=0)


    def _get_bootstrap_sample(self, X, y):
        """
        Returns a bootstrap sample of the same size as the original input X, 
        and the out-of-bag (oob) sample. According to the theoretical analysis, about 63.2% 
        of the original indexes will be included in the bootsrap sample. Some indexes will
        appear multiple times.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)                  The input samples.
        y : ndarray of shape (n_samples,)                             The target values.

        Returns
        -------
        bootstrap_sample_X : ndarray of shape (n_samples, n_features) The bootstrap sample of the input samples.
        bootstrap_sample_y : ndarray of shape (n_samples,)            The bootstrap sample of the target values.
        oob_sample_X : ndarray of shape (n_samples, n_features)       The out-of-bag sample of the input samples.
        oob_sample_y : ndarray of shape (n_samples,)                  The out-of-bag sample of the target values.
        """

        X, y = check_X_y(X, y)

        #TODO: ...
        n_samples = X.shape[0]
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        oob_idx = np.array(list(set(range(n_samples)) - set(bootstrap_idx)))

        bootstrap_sample_X = X[bootstrap_idx]
        bootstrap_sample_y = y[bootstrap_idx]
        oob_sample_X = X[oob_idx]
        oob_sample_y = y[oob_idx]

        return bootstrap_sample_X, bootstrap_sample_y, oob_sample_X, oob_sample_y

