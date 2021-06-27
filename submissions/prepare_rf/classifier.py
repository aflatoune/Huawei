from lightgbm import LGBMClassifier
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler


class Classifier:

    def __init__(self):
        self.clf = LGBMClassifier(
            n_estimators=2000, max_depth=-1, random_state=44, n_jobs=-1)
        print(self.clf)
        self.std = MinMaxScaler()

    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):

        X = np.vstack([X_source, X_target])
        y = np.hstack([y_source, y_target])

        selector = SelectFromModel(threshold='0.2*mean', estimator=LGBMClassifier()).fit(X, y)
        self.col_selected = selector.get_support()
        X = X[:, self.col_selected]
        X = self.std.fit_transform(X)
        self.clf.fit(X, y)

    def predict_proba(self, X_target, X_target_bkg):
      	X_target = X_target[:, self.col_selected]
      	X_target = self.std.transform(X_target)
        y_proba = self.clf.predict_proba(X_target)
        return y_proba


