import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier


def _preprocess_X(X_sparse):
    """CPM-like normalization + log1p, standard for scRNA-seq."""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)

def select_hvg(X, n_genes=2000):
    """Select genes with highest variance."""
    variances = np.var(X, axis=0)
    idx = np.argsort(variances)[-n_genes:]
    return idx

def _balance_classes(X_sparse, y):
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    idx_balanced = []

    for cls in unique_classes:
        idx_cls = np.where(y == cls)[0]
        selected_idx = np.random.choice(idx_cls, size=min_count, replace=False)
        idx_balanced.extend(selected_idx)

    idx_balanced = np.array(sorted(idx_balanced))
    return X_sparse[idx_balanced], y[idx_balanced]


class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        #self.le = LabelEncoder()
        self.pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=25),
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,        # profondeur des arbres de base
                subsample=0.8,      # comme "subsample" de xgboost (stochastic gradient boosting)
                random_state=42,
            )
        )

    def fit(self, X_sparse, y):
        # Balance classes
        X, y = _balance_classes(X_sparse, y)

        # Normalization
        X = _preprocess_X(X)

        # Reduction of the Noise
        self.hvg_idx_ = select_hvg(X, n_genes=2000)
        X = X[:, self.hvg_idx_]

        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_  

    def predict_proba(self, X_sparse):

        # Normalization
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]

        return self.pipe.predict_proba(X)