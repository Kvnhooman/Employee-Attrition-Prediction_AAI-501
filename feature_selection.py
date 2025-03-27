import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class FeatureSelection:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.selected_features = X.columns

    def variance_threshold(self, threshold=0.01):
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(self.X)
        self.selected_features = self.X.columns[selector.get_support()]
        return pd.DataFrame(X_selected, columns=self.selected_features)

    def correlation_selection(self, threshold=0.9):
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.selected_features = self.X.columns.difference(to_drop)
        return self.X[self.selected_features]

    def univariate_selection(self, k=5, method='f_classif'):
        score_func = f_classif if method == 'f_classif' else chi2
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(self.X, self.y)
        self.selected_features = self.X.columns[selector.get_support()]
        return pd.DataFrame(X_selected, columns=self.selected_features)

    def lasso_selection(self, alpha=0.01):
        model = Lasso(alpha=alpha)
        model.fit(self.X, self.y)
        self.selected_features = self.X.columns[model.coef_ != 0]
        return self.X[self.selected_features]

    def tree_based_selection(self, model=None, n_features=5):
        model = model or RandomForestClassifier()
        model.fit(self.X, self.y)
        feature_importances = model.feature_importances_
        top_features = np.argsort(feature_importances)[-n_features:]
        self.selected_features = self.X.columns[top_features]
        return self.X[self.selected_features]

    def get_selected_features(self):
        return list(self.selected_features)
