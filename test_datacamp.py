import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\flavi\OneDrive\Documents\ENSIIE\scmark-test")
from scipy.sparse import issparse

from problem import get_train_data, get_test_data
X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

# proportion de chaque type cellulaire, histogramme montrant si les classes sont équilibrées
lab_df = pd.DataFrame({'label': y_train})
lab_df.value_counts(normalize=True)
lab_df.label.hist()
print(X_train.shape)
print(type(X_train))

# label distribution
plt.figure(figsize=(6,4))
lab_df.label.value_counts().plot.bar()
plt.title("Distribution des types cellulaires dans le jeu d’entraînement")
plt.xlabel("Type de cellule")
plt.ylabel("Nombre d’observations")
plt.show()

# Densité des labels
print("Distribution normalisée des classes :")
print(lab_df.label.value_counts(normalize=True))


# --- Comptage total par gène (sans toarray) -----------------------------------

if issparse(X_train):
    total_genes_counts = np.array(X_train.sum(axis=0)).flatten()
else:
    total_genes_counts = X_train.sum(axis=0)

plt.figure(figsize=(6,4))
plt.hist(total_genes_counts, bins=50)
plt.xscale("log")
plt.title("Histogramme : total d'expression par gène (log-scale)")
plt.xlabel("Expression totale par gène (log)")
plt.ylabel("Nombre de gènes")
plt.show()

# --- Comptage total par cellule ----------------------------------------------

if issparse(X_train):
    total_cell_counts = np.array(X_train.sum(axis=1)).flatten()
else:
    total_cell_counts = X_train.sum(axis=1)

plt.figure(figsize=(6,4))
plt.hist(np.log10(total_cell_counts+1), bins=40)
plt.title("Histogramme : total d'expression par cellule (log10)")
plt.xlabel("log10(total d'ARN par cellule)")
plt.ylabel("Nombre de cellules")
plt.show()

X_train.toarray()

# combien chaque gène est exprimé au total
total_genes_counts = X_train.toarray().sum(axis=0)
# plt.hist(np.log10(total_genes_counts), bins = np.arange(6)) 
plt.hist(total_genes_counts, bins = 10**np.arange(6))
plt.xscale("log")
plt.title("Histogram of total gene (i.e. column) counts in log-scale.")
plt.xlabel('Total genes count (log-scale)')
plt.show()

# combien chaque cellule contient d’ARN total
total_cell_counts = X_train.toarray().sum(axis=1)
plt.hist(np.log10(total_cell_counts), bins = np.arange(1,6))
plt.title("Histogram of log-total cell (i.e. row) counts.")
plt.xlabel('log(cell_count)')
plt.show()



def preprocess_X(X):
    X = X.toarray()
    return X / X.sum(axis=1)[:, np.newaxis]

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


def compute_identical_counts(X):
    """
    Retourne, pour chaque colonne j, le nombre d'occurrences de la valeur la plus fréquente.
    """
    n_rows, n_cols = X.shape

    if issparse(X):
        nnz = np.array(X.getnnz(axis=0)).flatten()
        X_coo = X.tocoo()

        # dictionnaires pour compter les valeurs != 0
        counts = {j: {} for j in range(n_cols)}

        for i, j, v in zip(X_coo.row, X_coo.col, X_coo.data):
            counts[j][v] = counts[j].get(v, 0) + 1

        dominant_counts = np.zeros(n_cols)

        for j in range(n_cols):
            zero_count = n_rows - nnz[j]
            max_count = zero_count

            for v, c in counts[j].items():
                if c > max_count:
                    max_count = c

            dominant_counts[j] = max_count

        return dominant_counts

    else:
        dominant_counts = []
        for j in range(X.shape[1]):
            _, cnts = np.unique(X[:, j], return_counts=True)
            dominant_counts.append(cnts.max())
        return np.array(dominant_counts)

# ---- Utilisation ----
dominant_counts = compute_identical_counts(X_train)

plt.figure(figsize=(8,4))
plt.hist(dominant_counts, bins=50)
plt.xlabel("Nombre de valeurs identiques (valeur dominante)")
plt.ylabel("Nombre de gènes (variables)")
plt.title("Distribution du nombre de valeurs identiques par gène (avant filtrage)")
plt.show()

#def _preprocess_X(X_sparse):
#    # cast a dense array
#    X = X_sparse.toarray()

    # normalize each row
#    return X / X.sum(axis=1)[:, np.newaxis]

def compute_dominant_ratios(X):
    """
    Renvoie un tableau dominant_ratio[j] = proportion de la valeur la plus fréquente dans la colonne j.
    Compatible sparse.
    """
    import numpy as np
    from scipy.sparse import issparse

    n_rows, n_cols = X.shape

    if issparse(X):
        nnz = np.array(X.getnnz(axis=0)).flatten()
        X_coo = X.tocoo()

        counts = {j: {} for j in range(n_cols)}
        for i, j, v in zip(X_coo.row, X_coo.col, X_coo.data):
            counts[j][v] = counts[j].get(v, 0) + 1

        dominant_ratio = np.zeros(n_cols)

        for j in range(n_cols):
            zero_count = n_rows - nnz[j]
            max_count = zero_count

            for v, c in counts[j].items():
                if c > max_count:
                    max_count = c

            dominant_ratio[j] = max_count / n_rows

        return dominant_ratio

    else:
        # dense
        dominant_ratio = np.zeros(n_cols)
        for j in range(n_cols):
            vals, cnts = np.unique(X[:, j], return_counts=True)
            dominant_ratio[j] = cnts.max() / n_rows
        return dominant_ratio


def remove_highly_identical_columns(X, threshold=0.9):
    """
    Supprime les colonnes dont >= threshold des valeurs sont identiques (0 ou autre).
    Compatible sparse.
    """
    import numpy as np
    from scipy.sparse import issparse

    n_rows, n_cols = X.shape

    if issparse(X):
        nnz = np.array(X.getnnz(axis=0)).flatten()
        X_coo = X.tocoo()

        counts = {j: {} for j in range(n_cols)}

        for i, j, v in zip(X_coo.row, X_coo.col, X_coo.data):
            counts[j][v] = counts[j].get(v, 0) + 1

        dominant_ratio = np.zeros(n_cols)

        for j in range(n_cols):
            zero_count = n_rows - nnz[j]
            max_count = zero_count

            for v, c in counts[j].items():
                if c > max_count:
                    max_count = c

            dominant_ratio[j] = max_count / n_rows

        remove_mask = dominant_ratio >= threshold
        X_clean = X[:, ~remove_mask]

        return X_clean, remove_mask

    else:
        remove_mask = []
        for j in range(n_cols):
            vals, cnts = np.unique(X[:, j], return_counts=True)
            remove_mask.append(cnts.max() / n_rows >= threshold)

        remove_mask = np.array(remove_mask)
        X_clean = X[:, ~remove_mask]
        return X_clean, remove_mask



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

        # 1) ---- SUPPRESSION colonnes quasi constantes ----
        X_sparse, remove_mask = remove_highly_identical_columns(X_sparse, threshold=0.9)
        self.remove_mask_ = remove_mask  # pour utiliser dans predict
        # --- ANALYSE du filtrage ---

        print(f"Colonnes supprimées : {remove_mask.sum()} / {remove_mask.shape[0]}")
        print(f"Colonnes restantes : {remove_mask.shape[0] - remove_mask.sum()}")

        # Calcul du ratio dominant pour visualisation
        dominant_ratios = compute_dominant_ratios(X_sparse)

        plt.figure(figsize=(7,4))
        plt.hist(dominant_ratios, bins=30, color="steelblue")
        plt.axvline(0.9, color="red", linestyle="--", label="Seuil 90%")
        plt.title("Distribution des proportions de la valeur dominante par gène")
        plt.xlabel("Proportion de la valeur la plus fréquente")
        plt.ylabel("Nombre de gènes")
        plt.legend()
        plt.show()

        # Statistiques utiles
        print(f"Min ratio dominant : {dominant_ratios.min():.3f}")
        print(f"Max ratio dominant : {dominant_ratios.max():.3f}")
        print(f"Mean ratio dominant : {dominant_ratios.mean():.3f}")

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

        # 1) ---- Appliquer le MASQUE QUI SUPPRIME LES COLONNES ----
        X_sparse = X_sparse[:, ~self.remove_mask_]

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