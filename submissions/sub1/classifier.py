import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


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


class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        self.le = LabelEncoder()
        self.pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=25),
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",   # rapide et efficace
                random_state=42,
            ),
        )

    def fit(self, X_sparse, y):
        y_enc = self.le.fit_transform(y)
        # Normalization
        X = _preprocess_X(X_sparse)

        # Reduction of the Noise
        self.hvg_idx_ = select_hvg(X, n_genes=2000)
        X = X[:, self.hvg_idx_]

        self.pipe.fit(X, y_enc)
        #self.classes_ = self.pipe.classes_
        self.classes_ = self.le.classes_
        
        pass

    def predict_proba(self, X_sparse):

        # Normalization
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]

        return self.pipe.predict_proba(X)

    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        y_enc = np.argmax(proba, axis=1)
        return self.le.inverse_transform(y_enc)