from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Classifier:

    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=500, max_depth=-1, random_state=44, n_jobs=-1)
        print(self.clf)

    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):

        X = np.hstack([X_source, X_target])
        y = np.hstack([y_source, y_target])
        self.clf.fit(X, y)

    def predict_proba(self, X_target, X_target_bkg):
        y_proba = self.clf.predict_proba(X_target)
        return y_proba
