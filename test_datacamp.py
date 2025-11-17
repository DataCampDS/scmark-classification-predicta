import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\flavi\OneDrive\Documents\ENSIIE\scmark-test")


from problem import get_train_data, get_test_data
X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

lab_df = pd.DataFrame({'label': y_train})
lab_df.value_counts(normalize=True)


lab_df.label.hist()

print(X_train.shape)
print(type(X_train))

X_train.toarray()

total_genes_counts = X_train.toarray().sum(axis=0)
total_cell_counts = X_train.toarray().sum(axis=1)

# plt.hist(np.log10(total_genes_counts), bins = np.arange(6))
plt.hist(total_genes_counts, bins = 10**np.arange(6))
plt.xscale("log")
plt.title("Histogram of total gene (i.e. column) counts in log-scale.")
plt.xlabel('Total genes count (log-scale)')
plt.show()

plt.hist(np.log10(total_cell_counts), bins = np.arange(1,6))
plt.title("Histogram of log-total cell (i.e. row) counts.")
plt.xlabel('log(cell_count)')
plt.show()



#def preprocess_X(X):
#    X = X.toarray()
#    return X / X.sum(axis=1)[:, np.newaxis]

def preprocess_X(X):
    """CPM-like normalization + log1p, standard for scRNA-seq."""
    X = X.toarray()
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


# En modifiant le pre process

X_train_norm = preprocess_X(X_train)
# sanity check
np.allclose(X_train_norm.sum(axis=1), np.ones(X_train_norm.shape[0]))

from sklearn.metrics import balanced_accuracy_score

# this custom class is used by the challenge and calls 
# balanced_accuracy_score(y_true, y_pred, adjusted=False)
# under the hood
from problem import BalancedAccuracy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


pipe = Pipeline(
    [
        ("Scaler", StandardScaler(with_mean=True, with_std=True)),
        ("PCA with 50 components", PCA(n_components=50)),
        (
            "Random Forest Classifier",
            RandomForestClassifier(
                max_depth=5, n_estimators=100, max_features=3
            ),
        ),
    ]
)

pipe


# fit on train
pipe.fit(X_train_norm, y_train)
y_tr_pred = pipe.predict(X_train_norm)

# predict on test
X_test_norm = preprocess_X(X_test)
y_te_pred = pipe.predict(X_test_norm)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# compute balanced accuracy and confusion matrix
print(f"Train balanced accuracy : {balanced_accuracy_score(y_train, y_tr_pred):.3f}")
print(f"Test balanced accuracy : {balanced_accuracy_score(y_test, y_te_pred):.3f}")
cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_, )
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()



import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from umap import UMAP
import xgboost as xgb
from xgboost import XGBClassifier


#def _preprocess_X(X_sparse):
#    # cast a dense array
#    X = X_sparse.toarray()

    # normalize each row
#    return X / X.sum(axis=1)[:, np.newaxis]

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
            PCA(n_components=50),
            #RandomForestClassifier(
            #    max_depth=5, n_estimators=200, 
            #    max_features=10
            #),
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

        # PCA
        #self.reducer_ = PCA(n_components=50)
        #X = self.reducer_.fit_transform(X)
        #self.reducer_ = UMAP(n_components=30, random_state=0)
        #X = self.reducer_.fit_transform(X)

        self.pipe.fit(X, y_enc)
        #self.classes_ = self.pipe.classes_
        self.classes_ = self.le.classes_
        
        pass

    def predict_proba(self, X_sparse):

        # Normalization
        X = _preprocess_X(X_sparse)

        X = X[:, self.hvg_idx_]

        #X = self.reducer_.fit_transform(X)

        # here we use RandomForest.predict_proba()
        return self.pipe.predict_proba(X)
    
    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        y_enc = np.argmax(proba, axis=1)
        return self.le.inverse_transform(y_enc)



clf = Classifier()
clf.fit(X_train, y_train)
# predict_proba 
#y_tr_pred_proba = clf.predict_proba(X_train)
#y_te_pred_proba = clf.predict_proba(X_test)

# convert to hard classification with argmax
#y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
#y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

y_tr_pred = clf.predict(X_train)
y_te_pred = clf.predict(X_test)

print('Train balanced accuracy:', balanced_accuracy_score(y_train, y_tr_pred))
print('Test balanced accuracy:', balanced_accuracy_score(y_test, y_te_pred))

print(f"Train balanced accuracy : {balanced_accuracy_score(y_train, y_tr_pred):.3f}")
print(f"Test balanced accuracy : {balanced_accuracy_score(y_test, y_te_pred):.3f}")
cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_, )
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()