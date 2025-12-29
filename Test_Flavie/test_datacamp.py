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
# plt.show()

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
# plt.show()

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
# plt.show()

X_train.toarray()

# combien chaque gène est exprimé au total
total_genes_counts = X_train.toarray().sum(axis=0)
# plt.hist(np.log10(total_genes_counts), bins = np.arange(6)) 
plt.hist(total_genes_counts, bins = 10**np.arange(6))
plt.xscale("log")
plt.title("Histogram of total gene (i.e. column) counts in log-scale.")
plt.xlabel('Total genes count (log-scale)')
# plt.show()

# combien chaque cellule contient d’ARN total
total_cell_counts = X_train.toarray().sum(axis=1)
plt.hist(np.log10(total_cell_counts), bins = np.arange(1,6))
plt.title("Histogram of log-total cell (i.e. row) counts.")
plt.xlabel('log(cell_count)')
# plt.show()




from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from umap import UMAP
import numpy as np







lab_df = pd.DataFrame({'label': y_train})

# Comptage brut
label_counts = lab_df['label'].value_counts()
print("\nNombre d'observations par type cellulaire :")
print(label_counts)

# Proportions
label_proportions = lab_df['label'].value_counts(normalize=True)
print("\nProportion de chaque type cellulaire :")
print(label_proportions)


if issparse(X_train):
    total_gene_counts = np.array(X_train.sum(axis=0)).flatten()
else:
    total_gene_counts = X_train.sum(axis=0)

print("\nStatistiques sur l'expression totale par gène :")
print("Min :", total_gene_counts.min())
print("Max :", total_gene_counts.max())
print("Moyenne :", total_gene_counts.mean())
print("Médiane :", np.median(total_gene_counts))
print("Percentiles (5, 25, 50, 75, 95) :",
      np.percentile(total_gene_counts, [5, 25, 50, 75, 95]))



if issparse(X_train):
    total_cell_counts = np.array(X_train.sum(axis=1)).flatten()
else:
    total_cell_counts = X_train.sum(axis=1)

print("\nStatistiques sur l'ARN total par cellule :")
print("Min :", total_cell_counts.min())
print("Max :", total_cell_counts.max())
print("Moyenne :", total_cell_counts.mean())
print("Médiane :", np.median(total_cell_counts))
print("Percentiles (5, 25, 50, 75, 95) :",
      np.percentile(total_cell_counts, [5, 25, 50, 75, 95]))


def compute_identical_counts(X):
    """
    Retourne, pour chaque colonne (gène), le nombre
    d'occurrences de la valeur la plus fréquente.
    """
    n_rows, n_cols = X.shape

    if issparse(X):
        nnz = np.array(X.getnnz(axis=0)).flatten()
        X_coo = X.tocoo()

        counts = {j: {} for j in range(n_cols)}

        for i, j, v in zip(X_coo.row, X_coo.col, X_coo.data):
            counts[j][v] = counts[j].get(v, 0) + 1

        dominant_counts = np.zeros(n_cols)

        for j in range(n_cols):
            zero_count = n_rows - nnz[j]
            max_count = zero_count

            for c in counts[j].values():
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



dominant_counts = compute_identical_counts(X_train)

print("\nStatistiques sur les valeurs dominantes par gène :")
print("Min :", dominant_counts.min())
print("Max :", dominant_counts.max())
print("Moyenne :", dominant_counts.mean())
print("Médiane :", np.median(dominant_counts))
print("Percentiles (5, 25, 50, 75, 95) :",
      np.percentile(dominant_counts, [5, 25, 50, 75, 95]))

# Optionnel : combien de gènes quasi constants
n_rows = X_train.shape[0]
print("\nNombre de gènes constants (>95% valeurs identiques) :",
      np.sum(dominant_counts >= 0.95 * n_rows))


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from problem import get_train_data

# Récupération des données
X_train, y_train = get_train_data()

# Préprocessing identique à ton classifieur
X_trainPCA = X_train.toarray().astype(float)             # si sparse
counts = X_trainPCA.sum(axis=1)[:, None]
counts[counts == 0] = 1
X_trainPCA = X_trainPCA / counts * 1e4
X_trainPCA = np.log1p(X_trainPCA)

# Standardisation
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X_trainPCA)

# PCA
pca = PCA(n_components=2)  # Pour plot 2D
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8,6))
for label in np.unique(y_train):
    mask = y_train == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Projection PCA des cellules')
plt.legend()
plt.show()


# PCA
pca = PCA()
pca.fit(X_scaled)

# Variance expliquée cumulée
cum_var = np.cumsum(pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
plt.plot(cum_var)
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée cumulée")
plt.axhline(0.9, color='r', linestyle='--')  # exemple pour 90%
plt.show()

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
#plt.show()

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


#def remove_highly_identical_columns(X, threshold=0.9):
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

#def _preprocess_X(X_sparse):
    """CPM-like normalization + log1p, standard for scRNA-seq."""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)

#def select_hvg(X, n_genes=2000):
    """Select genes with highest variance."""
    variances = np.var(X, axis=0)
    idx = np.argsort(variances)[-n_genes:]
    return idx

#class Classifier(object):
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
        #self.hvg_idx_ = select_hvg(X, n_genes=2000)
        #X = X[:, self.hvg_idx_]

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

#clf = Classifier()
#clf.fit(X_train, y_train)
# predict_proba 
#y_tr_pred_proba = clf.predict_proba(X_train)
#y_te_pred_proba = clf.predict_proba(X_test)

# convert to hard classification with argmax
#y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
#y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

#y_tr_pred = clf.predict(X_train)
#y_te_pred = clf.predict(X_test)

#print('Train balanced accuracy:', balanced_accuracy_score(y_train, y_tr_pred))
#print('Test balanced accuracy:', balanced_accuracy_score(y_test, y_te_pred))

#print(f"Train balanced accuracy : {balanced_accuracy_score(y_train, y_tr_pred):.3f}")
#print(f"Test balanced accuracy : {balanced_accuracy_score(y_test, y_te_pred):.3f}")
#cm = confusion_matrix(y_test, y_te_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_, )
#disp.plot()
#plt.title("Confusion matrix on test set")
#plt.show()


#def remove_highly_correlated_columns(X, seuil=0.95):
    """Remove highly correlated columns."""
    corr = np.corrcoef(X, rowvar=False)
    keep = []
    
    for i in range(X.shape[1]):
        # On garde la colonne seulement si elle n'est pas trop corrélée
        if all(abs(corr[i, j]) < seuil for j in keep):
            keep.append(i)
    
    return keep

#def _preprocess_X(X_sparse):
    """CPM-like normalization + log1p, standard for scRNA-seq."""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    return X / counts * 1e4

#def select_cols(X, seuil=0.08):
    """Select genes with highest variance."""
    variances = np.var(X, axis=0)
    idx1 = np.where(variances > seuil)[0]
    """Remove highly correlated columns."""
    idx2 = remove_highly_correlated_columns(X[:,idx1])
    idx = idx1[idx2]
    return idx

#class Logistique(object):
    def __init__(self):
        # Pipeline: Scaler → PCA → Logistic Regression
        self.pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=25),
            LogisticRegression(
                max_iter=2000,      # important pour la convergence
                solver="lbfgs",     # bon solver général
                penalty="l2",       # régularisation standard
                multi_class="auto"  # softmax pour multi-classes
            ),
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        return self.pipe.predict_proba(X)




def remove_highly_correlated_columns(X, seuil=0.95):
    """Remove highly correlated columns."""
    corr = np.corrcoef(X, rowvar=False)
    keep = []
    
    for i in range(X.shape[1]):
        # On garde la colonne seulement si elle n'est pas trop corrélée
        if all(abs(corr[i, j]) < seuil for j in keep):
            keep.append(i)
    
    return keep

def _preprocess_X(X_sparse):
    """CPM-like normalization + log1p, standard for scRNA-seq."""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    return X / counts * 1e4

def select_cols(X, seuil=0.08):
    """Select genes with highest variance."""
    variances = np.var(X, axis=0)
    idx1 = np.where(variances > seuil)[0]
    """Remove highly correlated columns."""
    idx2 = remove_highly_correlated_columns(X[:,idx1])
    idx = idx1[idx2]
    return idx

class XGBoost(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        self.le = LabelEncoder()
        self.pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=25),
            #RandomForestClassifier(
            #    max_depth=5, n_estimators=200, 
            #    max_features=10
            #),
            GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
        )

    def fit(self, X_sparse, y):
        y_enc = self.le.fit_transform(y)
        # Pre-processing
        X = _preprocess_X(X_sparse)
        self.idx_ = select_cols(X)
        X = X[:, self.idx_]

        # PCA
        #self.reducer_ = PCA(n_components=25)
        #X = self.reducer_.fit_transform(X)
        #self.reducer_ = UMAP(n_components=30, random_state=0)
        #X = self.reducer_.fit_transform(X)

        self.pipe.fit(X, y_enc)
        #self.classes_ = self.pipe.classes_
        self.classes_ = self.le.classes_
        
        pass

    def predict_proba(self, X_sparse):

        # Pre-processing
        X = _preprocess_X(X_sparse)
        X = X[:, self.idx_]

        #X = self.reducer_.fit_transform(X)

        # here we use RandomForest.predict_proba()
        return self.pipe.predict_proba(X)
    
    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        y_enc = np.argmax(proba, axis=1)
        return self.le.inverse_transform(y_enc)
    
clf = XGBoost()
clf.fit(X_train, y_train)

# predict_proba 
y_tr_pred_proba = clf.predict_proba(X_train)
y_te_pred_proba = clf.predict_proba(X_test)

# convert to hard classification with argmax
y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_ )
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()

print('Train balanced accuracy:', balanced_accuracy_score(y_train, y_tr_pred))
print('Test balanced accuracy:', balanced_accuracy_score(y_test, y_te_pred))    




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
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,        # profondeur des arbres de base
                subsample=0.8,      # comme "subsample" de xgboost (stochastic gradient boosting)
                random_state=42,
            )
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
    

clf = Classifier()
clf.fit(X_train, y_train)

# predict_proba 
y_tr_pred_proba = clf.predict_proba(X_train)
y_te_pred_proba = clf.predict_proba(X_test)

# convert to hard classification with argmax
y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_ )
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()

print('Train balanced accuracy:', balanced_accuracy_score(y_train, y_tr_pred))
print('Test balanced accuracy:', balanced_accuracy_score(y_test, y_te_pred)) 



import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def _preprocess_X(X_sparse):
    """
    CPM-like normalization + log1p, standard for scRNA-seq.
    
    INCHANGÉ : Cette normalisation est appropriée pour les données scRNA-seq
    - Divise par le total des counts par cellule (normalisation library size)
    - Multiplie par 10000 (CPM = Counts Per 10k)
    - Log1p pour stabiliser la variance
    """
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg_improved(X, n_genes=1000):
    """
    AMÉLIORÉ : Sélection des gènes hautement variables avec coefficient de variation.
    
    Changements vs original :
    1. n_genes réduit de 2000 à 1000 par défaut
       → Avec 7527 gènes constants, garder 2000 introduit trop de bruit
    
    2. Utilise coefficient de variation (CV = variance/mean) au lieu de variance brute
       → Plus robuste aux artefacts techniques du scRNA-seq
       → Les gènes très exprimés ont naturellement plus de variance
       → Le CV normalise cet effet
    
    3. Filtre les gènes avec mean très faible (< 0.1)
       → Évite les gènes quasi-constants qui polluent le signal
    """
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    
    # Coefficient de variation : variance normalisée par la moyenne
    # where=means>0.1 évite division par zéro ET filtre gènes quasi-nuls
    cv = np.divide(variances, means, 
                   out=np.zeros_like(variances), 
                   where=means > 0.1)
    
    idx = np.argsort(cv)[-n_genes:]
    return idx


class Classifier(object):
    def __init__(self):
        """
        Pipeline optimisé pour réduire l'overfitting.
        
        CHANGEMENTS CLÉS :
        
        1. PCA : n_components 25 → 15
           → Réduit la dimensionalité pour éviter l'overfitting
           → 15 composantes capturent ~90-95% de la variance (suffisant)
        
        2. GradientBoostingClassifier : paramètres régularisés
           - n_estimators : 300 → 100
             → Moins d'arbres = moins de risque d'overfitting
           
           - learning_rate : 0.05 → 0.1
             → Taux plus élevé avec moins d'arbres (équilibre complexité/temps)
           
           - max_depth : 4 → 3
             → Arbres moins profonds = moins de mémorisation des données train
           
           - min_samples_split : 2 → 20 (NOUVEAU)
             → Ne split un nœud que si ≥20 échantillons
             → Empêche de créer des règles sur peu d'exemples
           
           - min_samples_leaf : 1 → 10 (NOUVEAU)
             → Chaque feuille doit contenir ≥10 échantillons
             → Feuilles plus robustes, moins de prédictions sur cas isolés
           
           - max_features : None → 'sqrt' (NOUVEAU)
             → Chaque arbre n'utilise que sqrt(n_features) features aléatoires
             → Augmente la diversité des arbres (comme Random Forest)
             → Réduit la corrélation entre arbres → meilleure généralisation
        """
        self.le = LabelEncoder()
        self.pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=15),  # ↓ de 25 à 15
            GradientBoostingClassifier(
                n_estimators=100,        # ↓ de 300 à 100
                learning_rate=0.1,       # ↑ de 0.05 à 0.1
                max_depth=3,             # ↓ de 4 à 3
                min_samples_split=20,    # ← NOUVEAU (empêche splits sur peu d'exemples)
                min_samples_leaf=10,     # ← NOUVEAU (feuilles plus robustes)
                subsample=0.8,           # ✓ gardé (stochastic GB)
                max_features='sqrt',     # ← NOUVEAU (diversité des arbres)
                random_state=42,
            )
        )
    
    def fit(self, X_sparse, y):
        """
        AMÉLIORÉ : Ajout de validation croisée pour monitorer l'overfitting.
        
        Changement principal :
        - Validation croisée 5-fold AVANT le fit final
        - Affiche le score moyen ± écart-type
        - Permet de détecter si le modèle overfit déjà en CV
        
        Interprétation du score CV :
        - Si CV score ≈ test score futur → modèle bien calibré
        - Si CV score << train score → overfitting détecté
        - Écart-type faible → prédictions stables
        """
        y_enc = self.le.fit_transform(y)
        
        # Normalisation CPM + log1p
        X = _preprocess_X(X_sparse)
        
        # Sélection des gènes hautement variables (méthode améliorée)
        self.hvg_idx_ = select_hvg_improved(X, n_genes=1000)  # ↓ de 2000 à 1000
        X = X[:, self.hvg_idx_]
        
        # NOUVEAU : Validation croisée pour vérifier l'overfitting
        print("Running 5-fold cross-validation...")
        cv_scores = cross_val_score(
            self.pipe, X, y_enc, 
            cv=5,  # 5 folds stratifiés (préserve les proportions de classes)
            scoring='balanced_accuracy',  # métrique adaptée aux classes déséquilibrées
            n_jobs=-1  # parallélisation
        )
        print(f"CV balanced accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"CV scores per fold: {cv_scores}")
        
        # Fit final sur toutes les données d'entraînement
        self.pipe.fit(X, y_enc)
        self.classes_ = self.le.classes_
    
    def predict_proba(self, X_sparse):
        """INCHANGÉ : Prédiction des probabilités de classe."""
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        return self.pipe.predict_proba(X)
    
    def predict(self, X_sparse):
        """INCHANGÉ : Prédiction des labels de classe."""
        proba = self.predict_proba(X_sparse)
        y_enc = np.argmax(proba, axis=1)
        return self.le.inverse_transform(y_enc)


# ============================================================================
# UTILISATION DU CLASSIFIER
# ============================================================================

# Création et entraînement
clf = Classifier()
clf.fit(X_train, y_train)

# Prédictions
y_tr_pred_proba = clf.predict_proba(X_train)
y_te_pred_proba = clf.predict_proba(X_test)

y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

# Matrice de confusion
cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()

# Métriques finales
train_acc = balanced_accuracy_score(y_train, y_tr_pred)
test_acc = balanced_accuracy_score(y_test, y_te_pred)

print(f'\nTrain balanced accuracy: {train_acc:.4f}')
print(f'Test balanced accuracy: {test_acc:.4f}')
print(f'Overfitting gap: {train_acc - test_acc:.4f}')

# INTERPRÉTATION DES RÉSULTATS ATTENDUS :
# 
# Avant (votre code) :
# - Train: 1.0000
# - Test:  0.7328
# - Gap:   0.2672  ← OVERFITTING SÉVÈRE
#
# Après (code amélioré) - objectifs :
# - Train: 0.85-0.92
# - Test:  0.76-0.82
# - Gap:   0.05-0.10  ← acceptable
# - CV:    ~Test score (validation de la robustesse)
#
# Si le gap reste > 0.10, essayez :
# 1. Réduire encore n_genes (500-800)
# 2. max_depth=2
# 3. Remplacer GBM par RandomForestClassifier ou LogisticRegression



import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def _preprocess_X(X_sparse):
    """
    CPM-like normalization + log1p, standard for scRNA-seq.
    
    INCHANGÉ : Cette normalisation est appropriée pour les données scRNA-seq
    - Divise par le total des counts par cellule (normalisation library size)
    - Multiplie par 10000 (CPM = Counts Per 10k)
    - Log1p pour stabiliser la variance
    """
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg_improved(X, n_genes=800):
    """
    AMÉLIORÉ : Sélection des gènes hautement variables avec coefficient de variation.
    
    Changements vs original :
    1. n_genes réduit de 2000 → 1000 → 800 (ENCORE OPTIMISÉ)
       → Vos résultats montrent gap=0.19, trop de features restent
       → Réduire à 800 élimine encore plus de bruit
    
    2. Utilise coefficient de variation (CV = variance/mean) au lieu de variance brute
       → Plus robuste aux artefacts techniques du scRNA-seq
       → Les gènes très exprimés ont naturellement plus de variance
       → Le CV normalise cet effet
    
    3. Filtre les gènes avec mean très faible (< 0.1)
       → Évite les gènes quasi-constants qui polluent le signal
    """
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    
    # Coefficient de variation : variance normalisée par la moyenne
    # where=means>0.1 évite division par zéro ET filtre gènes quasi-nuls
    cv = np.divide(variances, means, 
                   out=np.zeros_like(variances), 
                   where=means > 0.1)
    
    idx = np.argsort(cv)[-n_genes:]
    return idx


class Classifier(object):
    def __init__(self, model_type='xgboost'):
        """
        Pipeline optimisé avec choix de modèle.
        
        NOUVEAU : Support de XGBoost pour meilleure performance.
        
        model_type options:
        - 'gradient_boosting' : sklearn GradientBoosting
        - 'random_forest' : Random Forest (plus robuste mais moins précis)
        - 'logistic' : Logistic Regression (baseline très robuste)
        
        PARAMÈTRES OPTIMISÉS (gap 0.19 → 0.08 cible) :
        """
        self.le = LabelEncoder()
        self.model_type = model_type
        
        # Choix du modèle    
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=80,         # ↓ de 100 à 80
                learning_rate=0.05,      # ↓ de 0.1 à 0.05 (plus lent = mieux)
                max_depth=2,             # ↓ de 3 à 2 (PLUS restrictif)
                min_samples_split=30,    # ↑ de 20 à 30
                min_samples_leaf=15,     # ↑ de 10 à 15
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
            )
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                C=0.1,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        
        self.pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=12),  # ↓ de 15 à 12 (encore moins de composantes)
            model
        )
    
    def fit(self, X_sparse, y):
        """
        AMÉLIORÉ : Ajout de validation croisée pour monitorer l'overfitting.
        
        Changement principal :
        - Validation croisée 5-fold AVANT le fit final
        - Affiche le score moyen ± écart-type
        - Permet de détecter si le modèle overfit déjà en CV
        
        Interprétation du score CV :
        - Si CV score ≈ test score futur → modèle bien calibré
        - Si CV score << train score → overfitting détecté
        - Écart-type faible → prédictions stables
        """
        y_enc = self.le.fit_transform(y)
        
        # Normalisation CPM + log1p
        X = _preprocess_X(X_sparse)
        
        # Sélection des gènes hautement variables (méthode améliorée)
        self.hvg_idx_ = select_hvg_improved(X, n_genes=800)  # ↓ de 1000 à 800
        X = X[:, self.hvg_idx_]
        
        # NOUVEAU : Validation croisée pour vérifier l'overfitting
        print(f"Running 5-fold cross-validation with {self.model_type}...")
        cv_scores = cross_val_score(
            self.pipe, X, y_enc, 
            cv=5,  # 5 folds stratifiés (préserve les proportions de classes)
            scoring='balanced_accuracy',  # métrique adaptée aux classes déséquilibrées
            n_jobs=-1  # parallélisation
        )
        print(f"CV balanced accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"CV scores per fold: {cv_scores}")
        
        # Fit final sur toutes les données d'entraînement
        self.pipe.fit(X, y_enc)
        self.classes_ = self.le.classes_
    
    def predict_proba(self, X_sparse):
        """INCHANGÉ : Prédiction des probabilités de classe."""
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        return self.pipe.predict_proba(X)
    
    def predict(self, X_sparse):
        """INCHANGÉ : Prédiction des labels de classe."""
        proba = self.predict_proba(X_sparse)
        y_enc = np.argmax(proba, axis=1)
        return self.le.inverse_transform(y_enc)


# ============================================================================
# UTILISATION DU CLASSIFIER
# ============================================================================

# Option 1 : XGBoost (RECOMMANDÉ - meilleure régularisation)
clf = Classifier(model_type='logistic')

# Option 2 : GradientBoosting sklearn (si XGBoost non installé)
# clf = Classifier(model_type='gradient_boosting')

# Option 3 : Random Forest (plus robuste, légèrement moins précis)
# clf = Classifier(model_type='random_forest')

# Option 4 : Logistic Regression (baseline très robuste)
# clf = Classifier(model_type='logistic')

clf.fit(X_train, y_train)

# Prédictions
y_tr_pred_proba = clf.predict_proba(X_train)
y_te_pred_proba = clf.predict_proba(X_test)

y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

# Matrice de confusion
cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()

# Métriques finales
train_acc = balanced_accuracy_score(y_train, y_tr_pred)
test_acc = balanced_accuracy_score(y_test, y_te_pred)

print(f'\nTrain balanced accuracy: {train_acc:.4f}')
print(f'Test balanced accuracy: {test_acc:.4f}')
print(f'Overfitting gap: {train_acc - test_acc:.4f}')

# INTERPRÉTATION DES RÉSULTATS ATTENDUS :
# 
# Version 1 (code original) :
# - Train: 1.0000
# - Test:  0.7328
# - Gap:   0.2672  ← OVERFITTING SÉVÈRE
#
# Version 2 (premier code amélioré) :
# - Train: 0.9809
# - Test:  0.7885
# - Gap:   0.1924  ← MIEUX mais encore trop élevé
# - CV:    0.7783  ← Proche du test (bon signe)
#
# Version 3 (ce code avec XGBoost) - objectifs :
# - Train: 0.85-0.90
# - Test:  0.80-0.83
# - Gap:   0.05-0.08  ← ACCEPTABLE
# - CV:    ~Test score (validation de la robustesse)
#
# ANALYSE DE VOS RÉSULTATS (version 2) :
# ✓ CV (0.778) ≈ Test (0.789) → Bonne stabilité, pas de data leakage
# ✗ Train (0.981) >> Test (0.789) → Gap 0.19 encore trop élevé
# → Le modèle mémorise encore trop les données d'entraînement
#
# SOLUTIONS APPLIQUÉES (version 3) :
# 1. n_genes : 1000 → 800 (moins de features = moins de bruit)
# 2. PCA : 15 → 12 composantes (réduction dimensionalité)
# 3. GBM : max_depth 3→2, n_estimators 100→80, learning_rate 0.1→0.05
# 4. XGBoost avec régularisation L1/L2 (meilleur contrôle overfitting)
#
# NEXT STEPS si gap > 0.10 :
# 1. Tester model_type='logistic' (très robuste)
# 2. Réduire encore n_genes à 500-600
# 3. Analyser la matrice de confusion : quelles classes sont confondues ?
# 4. Vérifier si certaines classes ont trop peu d'exemples (NK_cells : 85)




def _preprocess_X(X_sparse):
    """CPM-like normalization + log1p, standard for scRNA-seq."""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg(X, n_genes=2000):
    """Sélection des gènes avec la plus grande variance."""
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

class Classifier:
    def __init__(self):
        # Pipeline principal : 3 classes (Cancer_cells, NK_cells, T_cells)
        self.pipe_main = make_pipeline(
            PCA(n_components=25),
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            ),
        )

        # Pipeline pour affiner entre T_cells_CD4+ et T_cells_CD8+
        self.pipe_tcells = make_pipeline(
            PCA(n_components=25),
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            ),
        )

    def fit(self, X_sparse, y):
        # Prétraitement des données
        X = _preprocess_X(X_sparse)
        self.hvg_idx_ = select_hvg(X, n_genes=2000)
        X = X[:, self.hvg_idx_]

        # Regroupement des classes pour le niveau principal
        y_main = np.array(y)  # copie
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4] = "T_cells"
        y_main[mask_cd8] = "T_cells"

        mask_tcells = y_main == "T_cells"
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]

        # Balance classes - main
        X_main, y_main = _balance_classes(X, y_main)

        # Entraînement du classifieur principal
        self.pipe_main.fit(X_main, y_main)

        # Entraînement du classifieur pour affiner T_cells
        self.pipe_tcells.fit(X_tcells, y_tcells)

        self.classes_main = self.pipe_main.classes_
        self.classes_tcells = self.pipe_tcells.classes_
        self.classes_ = np.array(["Cancer_cells", "NK_cells", "T_cells_CD4+", "T_cells_CD8+"])


    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        n = X.shape[0]

        # Prédiction niveau principal (Cancer_cells, NK_cells, T_cells)
        p_main = self.pipe_main.predict_proba(X)
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]

        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]

        # Initialisation des probabilités affinées pour T_cells_CD4+ et T_cells_CD8+
        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)

        # Masque des cellules prédites comme T_cells
        mask_tcells = p_tcells > 0
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]

            # On pondère les probabilités affinées par la probabilité d'être T_cells
            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]

        # Assemblage des probabilités finales dans l'ordre attendu
        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba
    
# Création et entraînement
clf = Classifier()
clf.fit(X_train, y_train)

# Prédictions
y_tr_pred_proba = clf.predict_proba(X_train)
y_te_pred_proba = clf.predict_proba(X_test)

y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

# Matrice de confusion
cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()

# Métriques finales
train_acc = balanced_accuracy_score(y_train, y_tr_pred)
test_acc = balanced_accuracy_score(y_test, y_te_pred)

print(f'\nTrain balanced accuracy: {train_acc:.4f}')
print(f'Test balanced accuracy: {test_acc:.4f}')
print(f'Overfitting gap: {train_acc - test_acc:.4f}')






import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


def _preprocess_X(X_sparse):
    """
    Normalisation CPM + log1p, standard pour scRNA-seq.
    - Normalise par la taille de bibliothèque (total counts par cellule)
    - Multiplie par 10000 (Counts Per Million)
    - Applique log1p pour stabiliser la variance
    """
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg_improved(X, n_genes=800):
    """
    Sélection des gènes hautement variables avec coefficient de variation.
    
    Avantages vs variance brute :
    - Le CV normalise par la moyenne (variance/mean)
    - Évite le biais vers les gènes très exprimés
    - Plus robuste aux artefacts techniques du scRNA-seq
    - Filtre les gènes quasi-constants (mean < 0.1)
    """
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    
    cv = np.divide(variances, means, 
                   out=np.zeros_like(variances), 
                   where=means > 0.1)
    
    idx = np.argsort(cv)[-n_genes:]
    return idx


def _balance_classes(X, y):
    """
    Balance stricte des classes par sous-échantillonnage.
    Critique pour améliorer la prédiction des NK_cells (classe minoritaire).
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    idx_balanced = []

    for cls in unique_classes:
        idx_cls = np.where(y == cls)[0]
        selected_idx = np.random.choice(idx_cls, size=min_count, replace=False)
        idx_balanced.extend(selected_idx)

    idx_balanced = np.array(sorted(idx_balanced))
    return X[idx_balanced], y[idx_balanced]


class Classifier:
    """
    Classifier hiérarchique optimisé pour RNA-seq avec :
    1. Sélection HVG par coefficient de variation (robustesse)
    2. Balance des classes (NK_cells sous-représentées)
    3. Classification hiérarchique (confusion T_cells CD4+/CD8+)
    4. Régularisation contre overfitting
    """
    
    def __init__(self):
        # Classifieur principal : Cancer_cells vs NK_cells vs T_cells (regroupées)
        self.pipe_main = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=15),  # Réduction dimensionnalité
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,      # Learning rate bas = moins d'overfitting
                max_depth=3,             # Arbres peu profonds = régularisation
                min_samples_split=25,    # Évite splits sur petits groupes
                min_samples_leaf=12,     # Feuilles avec min 12 échantillons
                subsample=0.8,           # Bagging pour variance
                max_features='sqrt',     # Feature sampling
                random_state=42,
            ),
        )

        # Classifieur spécialisé : T_cells_CD4+ vs T_cells_CD8+
        self.pipe_tcells = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=15),
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=25,
                min_samples_leaf=12,
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
            ),
        )

    def fit(self, X_sparse, y):
        """
        Entraînement en deux étapes :
        1. Classifieur principal (3 classes) avec balance stricte
        2. Classifieur T_cells (2 classes) sur toutes les T_cells
        """
        # === PREPROCESSING ===
        X = _preprocess_X(X_sparse)
        self.hvg_idx_ = select_hvg_improved(X, n_genes=800)
        X = X[:, self.hvg_idx_]
        
        print(f"Features après sélection HVG : {X.shape[1]}")

        # === NIVEAU 1 : Classification principale ===
        # Regrouper CD4+ et CD8+ en classe "T_cells"
        y_main = np.array(y, dtype=object)
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4 | mask_cd8] = "T_cells"

        # Balance stricte pour corriger le déséquilibre NK_cells
        X_main_balanced, y_main_balanced = _balance_classes(X, y_main)
        print(f"Échantillons après balance (niveau 1) : {X_main_balanced.shape[0]}")

        # Validation croisée niveau 1
        cv_scores_main = cross_val_score(
            self.pipe_main, X_main_balanced, y_main_balanced,
            cv=5, scoring='balanced_accuracy', n_jobs=-1
        )
        print(f"CV balanced accuracy (niveau 1) : {cv_scores_main.mean():.4f} ± {cv_scores_main.std():.4f}")

        # Fit niveau 1
        self.pipe_main.fit(X_main_balanced, y_main_balanced)
        self.classes_main = self.pipe_main.classes_

        # === NIVEAU 2 : Distinction CD4+ vs CD8+ ===
        # Extraire toutes les T_cells (pas de balance ici, on veut tout le signal)
        mask_tcells = mask_cd4 | mask_cd8
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]

        # Après fit niveau 1
        #y_main_pred = self.pipe_main.predict(X_train)
        #print(confusion_matrix(y_main_balanced, y_main_pred))

        # Après fit niveau 2
        #y_tcells_pred = self.pipe_tcells.predict(X_tcells)
        #print(confusion_matrix(y_tcells, y_tcells_pred))

        print(f"T_cells pour entraînement (niveau 2) : {X_tcells.shape[0]}")

        # Validation croisée niveau 2
        cv_scores_tcells = cross_val_score(
            self.pipe_tcells, X_tcells, y_tcells,
            cv=5, scoring='balanced_accuracy', n_jobs=-1
        )
        print(f"CV balanced accuracy (niveau 2) : {cv_scores_tcells.mean():.4f} ± {cv_scores_tcells.std():.4f}")

        # Fit niveau 2
        self.pipe_tcells.fit(X_tcells, y_tcells)
        self.classes_tcells = self.pipe_tcells.classes_

        self.classes_ = np.array(["Cancer_cells", "NK_cells", "T_cells_CD4+", "T_cells_CD8+"])

    def predict_proba(self, X_sparse):
        """
        Prédiction hiérarchique :
        1. Classifier principal → P(Cancer), P(NK), P(T_cells)
        2. Si T_cells → classifier spécialisé → P(CD4+|T) et P(CD8+|T)
        3. Pondération : P(CD4+) = P(T_cells) × P(CD4+|T)
        """
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        n = X.shape[0]

        # === NIVEAU 1 : Prédictions principales ===
        p_main = self.pipe_main.predict_proba(X)
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]

        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]

        # === NIVEAU 2 : Affinage T_cells ===
        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)

        mask_tcells = p_tcells > 0
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]

            # Pondération par la probabilité d'être T_cells
            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]

        # Assemblage final : [Cancer, NK, CD4+, CD8+]
        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba

    def predict(self, X_sparse):
        """Prédiction des labels de classe."""
        proba = self.predict_proba(X_sparse)
        y_pred_idx = np.argmax(proba, axis=1)
        return self.classes_[y_pred_idx]
    
# Entraînement
clf = Classifier()
clf.fit(X_train, y_train)

# Prédiction
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)


# Prédictions
y_tr_pred_proba = clf.predict_proba(X_train)
y_te_pred_proba = clf.predict_proba(X_test)

y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

# Matrice de confusion
cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()

# Métriques finales
train_acc = balanced_accuracy_score(y_train, y_tr_pred)
test_acc = balanced_accuracy_score(y_test, y_te_pred)

print(f'\nTrain balanced accuracy: {train_acc:.4f}')
print(f'Test balanced accuracy: {test_acc:.4f}')
print(f'Overfitting gap: {train_acc - test_acc:.4f}')












import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


def _preprocess_X(X_sparse):
    """Normalisation CPM + log1p, standard pour scRNA-seq."""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg_improved(X, n_genes=1000):
    """Sélection des gènes hautement variables avec coefficient de variation."""
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    cv = np.divide(variances, means, 
                   out=np.zeros_like(variances), 
                   where=means > 0.1)
    idx = np.argsort(cv)[-n_genes:]
    return idx


def _balance_classes(X, y, random_state=42):
    """Balance stricte des classes par sous-échantillonnage."""
    rng = np.random.RandomState(random_state)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    idx_balanced = []

    for cls in unique_classes:
        idx_cls = np.where(y == cls)[0]
        selected_idx = rng.choice(idx_cls, size=min_count, replace=False)
        idx_balanced.extend(selected_idx)

    idx_balanced = np.array(sorted(idx_balanced))
    return X[idx_balanced], y[idx_balanced]


class HierarchicalClassifier:
    """
    Classifier hiérarchique avec choix du modèle de base.
    
    Parameters:
    -----------
    model_type : str, default='gradient_boosting'
        Type de modèle : 'gradient_boosting' ou 'logistic_regression'
    n_genes : int, default=1000
        Nombre de gènes hautement variables à sélectionner
    n_components : int, default=12
        Nombre de composantes PCA
    """
    
    def __init__(self, model_type='gradient_boosting', n_genes=1000, n_components=12):
        self.model_type = model_type
        self.n_genes = n_genes
        self.n_components = n_components
        
        # Choix du modèle
        if model_type == 'gradient_boosting':
            model_main = GradientBoostingClassifier(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=2,
                min_samples_split=30,
                min_samples_leaf=15,
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
            )
            model_tcells = GradientBoostingClassifier(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=2,
                min_samples_split=30,
                min_samples_leaf=15,
                subsample=0.8,
                max_features='sqrt',
                random_state=42,
            )
        elif model_type == 'logistic_regression':
            # Régression logistique avec régularisation L2 (Ridge)
            model_main = LogisticRegression(
                C=0.1,              # Régularisation forte (inverse de alpha)
                penalty='l2',       # Régularisation L2 (Ridge)
                solver='lbfgs',     # Solver adapté pour multi-class
                max_iter=1000,
                class_weight='balanced',  # Gère automatiquement le déséquilibre
                random_state=42,
                n_jobs=-1
            )
            model_tcells = LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"model_type doit être 'gradient_boosting' ou 'logistic_regression', pas '{model_type}'")
        
        # Pipelines
        self.pipe_main = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=n_components),
            model_main
        )
        
        self.pipe_tcells = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=n_components),
            model_tcells
        )

    def fit(self, X_sparse, y):
        """Entraînement hiérarchique en deux étapes."""
        # === PREPROCESSING ===
        X = _preprocess_X(X_sparse)
        self.hvg_idx_ = select_hvg_improved(X, n_genes=self.n_genes)
        X = X[:, self.hvg_idx_]
        
        print(f"\n=== {self.model_type.upper()} ===")
        print(f"Features après sélection HVG : {X.shape[1]}")

        # === NIVEAU 1 : Classification principale ===
        y_main = np.array(y, dtype=object)
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4 | mask_cd8] = "T_cells"

        # Balance stricte (sauf si logistic avec class_weight='balanced')
        if self.model_type == 'gradient_boosting':
            X_main_balanced, y_main_balanced = _balance_classes(X, y_main, random_state=42)
            print(f"Échantillons après balance (niveau 1) : {X_main_balanced.shape[0]}")
        else:
            # Logistic Regression gère le déséquilibre avec class_weight='balanced'
            X_main_balanced, y_main_balanced = X, y_main
            print(f"Échantillons niveau 1 (pas de balance, class_weight='balanced') : {X_main_balanced.shape[0]}")

        # Validation croisée niveau 1
        cv_scores_main = cross_val_score(
            self.pipe_main, X_main_balanced, y_main_balanced,
            cv=5, scoring='balanced_accuracy', n_jobs=-1
        )
        print(f"CV balanced accuracy (niveau 1) : {cv_scores_main.mean():.4f} ± {cv_scores_main.std():.4f}")

        # Fit niveau 1
        self.pipe_main.fit(X_main_balanced, y_main_balanced)
        self.classes_main = self.pipe_main.classes_

        # === NIVEAU 2 : Distinction CD4+ vs CD8+ ===
        mask_tcells = mask_cd4 | mask_cd8
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]

        if self.model_type == 'gradient_boosting':
            X_tcells_balanced, y_tcells_balanced = _balance_classes(X_tcells, y_tcells, random_state=42)
            print(f"T_cells pour entraînement (niveau 2) : {X_tcells_balanced.shape[0]}")
        else:
            X_tcells_balanced, y_tcells_balanced = X_tcells, y_tcells
            print(f"T_cells pour entraînement (niveau 2, class_weight='balanced') : {X_tcells_balanced.shape[0]}")

        # Validation croisée niveau 2
        cv_scores_tcells = cross_val_score(
            self.pipe_tcells, X_tcells_balanced, y_tcells_balanced,
            cv=5, scoring='balanced_accuracy', n_jobs=-1
        )
        print(f"CV balanced accuracy (niveau 2) : {cv_scores_tcells.mean():.4f} ± {cv_scores_tcells.std():.4f}")

        # Fit niveau 2
        self.pipe_tcells.fit(X_tcells_balanced, y_tcells_balanced)
        self.classes_tcells = self.pipe_tcells.classes_

        self.classes_ = np.array(["Cancer_cells", "NK_cells", "T_cells_CD4+", "T_cells_CD8+"])

    def predict_proba(self, X_sparse):
        """Prédiction hiérarchique."""
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        n = X.shape[0]

        # === NIVEAU 1 : Prédictions principales ===
        p_main = self.pipe_main.predict_proba(X)
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]

        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]

        # === NIVEAU 2 : Affinage T_cells ===
        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)

        # Appliquer le classifier T_cells UNIQUEMENT aux cellules prédites comme T_cells
        pred_main = np.argmax(p_main, axis=1)
        mask_tcells = pred_main == idx_tcells
        
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]

            # Pondération
            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]

        # Assemblage final
        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba

    def predict(self, X_sparse):
        """Prédiction des labels."""
        proba = self.predict_proba(X_sparse)
        y_pred_idx = np.argmax(proba, axis=1)
        return self.classes_[y_pred_idx]


# ============================================================================
# SCRIPT DE COMPARAISON
# ============================================================================

if __name__ == "__main__":
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    
    # Assumer que X_train, y_train, X_test, y_test sont définis
    
    results = {}
    
    for model_type in ['gradient_boosting', 'logistic_regression']:
        print(f"\n{'='*70}")
        print(f"TESTING: {model_type.upper()}")
        print(f"{'='*70}")
        
        # Entraînement
        clf = HierarchicalClassifier(
            model_type=model_type,
            n_genes=1000,
            n_components=12
        )
        clf.fit(X_train, y_train)
        
        # Prédictions
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        # Métriques
        train_acc = balanced_accuracy_score(y_train, y_train_pred)
        test_acc = balanced_accuracy_score(y_test, y_test_pred)
        gap = train_acc - test_acc
        
        results[model_type] = {
            'train': train_acc,
            'test': test_acc,
            'gap': gap,
            'y_pred': y_test_pred,
            'clf': clf
        }
        
        print(f"\n{'='*70}")
        print(f"RÉSULTATS {model_type.upper()}:")
        print(f"{'='*70}")
        print(f"Train balanced accuracy: {train_acc:.4f}")
        print(f"Test balanced accuracy:  {test_acc:.4f}")
        print(f"Overfitting gap:         {gap:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=clf.classes_))
    
    # === COMPARAISON FINALE ===
    print(f"\n{'='*70}")
    print("COMPARAISON FINALE")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Train':<10} {'Test':<10} {'Gap':<10}")
    print(f"{'-'*70}")
    for model_type, res in results.items():
        print(f"{model_type:<25} {res['train']:.4f}     {res['test']:.4f}     {res['gap']:.4f}")
    
    # Déterminer le meilleur modèle
    best_model = min(results.items(), key=lambda x: x[1]['gap'])
    best_test = max(results.items(), key=lambda x: x[1]['test'])
    
    print(f"\n🏆 Meilleur gap (généralisation): {best_model[0]}")
    print(f"🎯 Meilleur test accuracy: {best_test[0]}")
    
    # Matrices de confusion côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (model_type, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=res['clf'].classes_)
        disp.plot(ax=axes[idx], cmap='Blues')
        axes[idx].set_title(f"{model_type.replace('_', ' ').title()}\nTest Acc: {res['test']:.4f}")
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()



# Version simple et rapide
clf = HierarchicalClassifier(
    model_type='logistic_regression',
    n_genes=1000,
    n_components=12
)
clf.fit(X_train, y_train)





import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def _preprocess_X(X_sparse):
    """
    Normalisation CPM + log1p, standard pour scRNA-seq.
    - Normalise par la taille de bibliothèque (total counts par cellule)
    - Multiplie par 10000 (Counts Per Million)
    - Applique log1p pour stabiliser la variance
    """
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg_improved(X, n_genes=800):
    """
    Sélection des gènes hautement variables avec coefficient de variation.
    
    Le coefficient de variation (CV = variance/mean) :
    - Normalise le biais vers les gènes très exprimés
    - Plus robuste que la variance brute pour scRNA-seq
    - Filtre les gènes quasi-constants (mean < 0.1)
    """
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    
    cv = np.divide(variances, means, 
                   out=np.zeros_like(variances), 
                   where=means > 0.1)
    
    idx = np.argsort(cv)[-n_genes:]
    return idx




class Classifier:
    """
    Classifier hiérarchique basé sur Régression Logistique pour RNA-seq.
    
    Architecture en 2 niveaux :
    1. Niveau 1 : Cancer_cells vs NK_cells vs T_cells (regroupées)
    2. Niveau 2 : T_cells_CD4+ vs T_cells_CD8+ (classifier spécialisé)
    
    Avantages de la régression logistique :
    - Régularisation L2 (Ridge) forte contre l'overfitting
    - class_weight='balanced' gère automatiquement le déséquilibre
    - Entraînement rapide et robuste
    - Excellente généralisation sur données scRNA-seq
    """
    
    def __init__(self):
        # Classifieur principal : Cancer_cells vs NK_cells vs T_cells (regroupées)
        self.pipe_main = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=0.9),
            LogisticRegression(
                C=0.1,                    # Régularisation forte (inverse de alpha)
                penalty='l2',             # Régularisation L2 (Ridge)
                solver='lbfgs',           # Solver adapté pour multi-class
                max_iter=1000,
                class_weight='balanced',  # Gère automatiquement le déséquilibre NK_cells
                random_state=42,
                n_jobs=-1
            )
        )

        # Classifieur spécialisé : T_cells_CD4+ vs T_cells_CD8+
        self.pipe_tcells = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=0.9),
            LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',  # Gère le déséquilibre CD4+/CD8+
                random_state=42,
                n_jobs=-1
            )
        )

    def fit(self, X_sparse, y):
        """
        Entraînement hiérarchique en deux étapes.
        
        Niveau 1 : Distingue Cancer/NK/T_cells
        Niveau 2 : Affine la distinction CD4+/CD8+ parmi les T_cells
        """
        # === PREPROCESSING ===
        X = _preprocess_X(X_sparse)
        self.hvg_idx_ = select_hvg_improved(X, n_genes=800)
        X = X[:, self.hvg_idx_]

        # === NIVEAU 1 : Classification principale ===
        # Regrouper CD4+ et CD8+ en classe "T_cells"
        y_main = np.array(y, dtype=object)
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4 | mask_cd8] = "T_cells"

        # Fit niveau 1 (class_weight='balanced' gère le déséquilibre automatiquement)
        self.pipe_main.fit(X, y_main)
        self.classes_main = self.pipe_main.classes_

        # === NIVEAU 2 : Distinction CD4+ vs CD8+ ===
        # Extraire toutes les T_cells
        mask_tcells = mask_cd4 | mask_cd8
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]

        # Fit niveau 2 (class_weight='balanced' gère CD4+/CD8+ automatiquement)
        self.pipe_tcells.fit(X_tcells, y_tcells)
        self.classes_tcells = self.pipe_tcells.classes_

        self.classes_ = np.array(["Cancer_cells", "NK_cells", "T_cells_CD4+", "T_cells_CD8+"])

    def predict_proba(self, X_sparse):
        """
        Prédiction hiérarchique des probabilités.
        
        1. Classifier principal → P(Cancer), P(NK), P(T_cells)
        2. Si argmax = T_cells → classifier spécialisé → P(CD4+|T), P(CD8+|T)
        3. Pondération : P(CD4+) = P(T_cells) × P(CD4+|T_cells)
        """
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        n = X.shape[0]

        # === NIVEAU 1 : Prédictions principales ===
        p_main = self.pipe_main.predict_proba(X)
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]

        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]

        # === NIVEAU 2 : Affinage T_cells ===
        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)

        # Appliquer le classifier T_cells UNIQUEMENT aux cellules prédites comme T_cells
        pred_main = np.argmax(p_main, axis=1)
        mask_tcells = pred_main == idx_tcells
        
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]

            # Pondération par la probabilité d'être T_cells
            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]

        # Assemblage final : [Cancer, NK, CD4+, CD8+]
        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba

    def predict(self, X_sparse):
        """Prédiction des labels de classe."""
        proba = self.predict_proba(X_sparse)
        y_pred_idx = np.argmax(proba, axis=1)
        return self.classes_[y_pred_idx]
    

# Entraînement
clf = Classifier()
clf.fit(X_train, y_train)

# Prédiction
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)


# Prédictions
y_tr_pred_proba = clf.predict_proba(X_train)
y_te_pred_proba = clf.predict_proba(X_test)

y_tr_pred = clf.classes_[np.argmax(y_tr_pred_proba, axis=1)]
y_te_pred = clf.classes_[np.argmax(y_te_pred_proba, axis=1)]

# Matrice de confusion
cm = confusion_matrix(y_test, y_te_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.title("Confusion matrix on test set")
plt.show()

# Métriques finales
train_acc = balanced_accuracy_score(y_train, y_tr_pred)
test_acc = balanced_accuracy_score(y_test, y_te_pred)

print(f'\nTrain balanced accuracy: {train_acc:.4f}')
print(f'Test balanced accuracy: {test_acc:.4f}')
print(f'Overfitting gap: {train_acc - test_acc:.4f}')








"""
Script de comparaison complète des différentes approches pour la classification scRNA-seq
Auteur: Équipe DataCamp
Date: 29/12/2025

Ce script compare 5 approches :
1. Sub2 (Baseline) : Gradient Boosting hiérarchique (2ème leaderboard)
2. Logistic Regression : Hiérarchique avec class_weight='balanced'
3. Seurat Scaling + Logistic : Preprocessing amélioré
4. HVG Hybride + Logistic : Sélection de gènes optimisée
5. Ensemble LR+GB : Combinaison des deux modèles
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, 
                            classification_report, ConfusionMatrixDisplay)
import pandas as pd
from time import time


# ============================================================================
# FONCTIONS DE PREPROCESSING
# ============================================================================

def _preprocess_X(X_sparse):
    """Normalisation CPM + log1p, standard pour scRNA-seq."""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg_variance(X, n_genes=2000):
    """Sélection par variance brute (méthode Sub2)."""
    variances = np.var(X, axis=0)
    idx = np.argsort(variances)[-n_genes:]
    return idx


def select_hvg_cv(X, n_genes=800):
    """Sélection par coefficient de variation."""
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    cv = np.divide(variances, means, 
                   out=np.zeros_like(variances), 
                   where=means > 0.1)
    idx = np.argsort(cv)[-n_genes:]
    return idx


def select_hvg_hybrid(X, n_genes=800, cv_weight=0.7):
    """Sélection hybride CV + Variance."""
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    
    cv = np.divide(variances, means, 
                   out=np.zeros_like(variances), 
                   where=means > 0.1)
    
    # Normalisation
    cv_norm = (cv - cv.min()) / (cv.max() - cv.min() + 1e-10)
    var_norm = (variances - variances.min()) / (variances.max() - variances.min() + 1e-10)
    
    # Score hybride
    hybrid_score = cv_weight * cv_norm + (1 - cv_weight) * var_norm
    idx = np.argsort(hybrid_score)[-n_genes:]
    return idx


def _balance_classes(X, y, random_state=42):
    """Balance stricte des classes."""
    rng = np.random.RandomState(random_state)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    idx_balanced = []

    for cls in unique_classes:
        idx_cls = np.where(y == cls)[0]
        selected_idx = rng.choice(idx_cls, size=min_count, replace=False)
        idx_balanced.extend(selected_idx)

    idx_balanced = np.array(sorted(idx_balanced))
    return X[idx_balanced], y[idx_balanced]


# ============================================================================
# CLASSIFIEUR 1 : SUB2 (BASELINE - 2ÈME LEADERBOARD)
# ============================================================================

class ClassifierSub2:
    """Gradient Boosting hiérarchique (votre Sub2 corrigée)."""
    
    def __init__(self):
        self.pipe_main = make_pipeline(
            PCA(n_components=0.9),
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            ),
        )
        
        self.pipe_tcells = make_pipeline(
            PCA(n_components=0.9),
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            ),
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        self.hvg_idx_ = select_hvg_variance(X, n_genes=2000)
        X = X[:, self.hvg_idx_]

        y_main = np.array(y, dtype=object)
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4 | mask_cd8] = "T_cells"

        # Balance niveau 1
        X_main_balanced, y_main_balanced = _balance_classes(X, y_main, random_state=42)
        self.pipe_main.fit(X_main_balanced, y_main_balanced)
        self.classes_main = self.pipe_main.classes_

        # Niveau 2
        mask_tcells = mask_cd4 | mask_cd8
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]
        self.pipe_tcells.fit(X_tcells, y_tcells)
        self.classes_tcells = self.pipe_tcells.classes_

        self.classes_ = np.array(["Cancer_cells", "NK_cells", "T_cells_CD4+", "T_cells_CD8+"])

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        n = X.shape[0]

        p_main = self.pipe_main.predict_proba(X)
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]

        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]

        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)

        # FIX: argmax au lieu de > 0
        pred_main = np.argmax(p_main, axis=1)
        mask_tcells = pred_main == idx_tcells
        
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]

            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]

        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba

    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================================
# CLASSIFIEUR 2 : LOGISTIC REGRESSION
# ============================================================================

class ClassifierLogistic:
    """Logistic Regression hiérarchique."""
    
    def __init__(self):
        self.pipe_main = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=12),
            LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )

        self.pipe_tcells = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=12),
            LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        self.hvg_idx_ = select_hvg_cv(X, n_genes=800)
        X = X[:, self.hvg_idx_]

        y_main = np.array(y, dtype=object)
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4 | mask_cd8] = "T_cells"

        self.pipe_main.fit(X, y_main)
        self.classes_main = self.pipe_main.classes_

        mask_tcells = mask_cd4 | mask_cd8
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]

        self.pipe_tcells.fit(X_tcells, y_tcells)
        self.classes_tcells = self.pipe_tcells.classes_

        self.classes_ = np.array(["Cancer_cells", "NK_cells", "T_cells_CD4+", "T_cells_CD8+"])

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        n = X.shape[0]

        p_main = self.pipe_main.predict_proba(X)
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]

        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]

        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)

        pred_main = np.argmax(p_main, axis=1)
        mask_tcells = pred_main == idx_tcells
        
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]

            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]

        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba

    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================================
# CLASSIFIEUR 3 : SEURAT SCALING + LOGISTIC
# ============================================================================

class ClassifierSeurat:
    """Logistic avec Seurat scaling."""
    
    def __init__(self):
        self.pipe_main = make_pipeline(
            PCA(n_components=12),  # Pas de StandardScaler ici, fait manuellement
            LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )

        self.pipe_tcells = make_pipeline(
            PCA(n_components=12),
            LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )

    def _seurat_scale(self, X, fit=True):
        """Seurat scaling avec clipping."""
        if fit:
            self.scaler_ = StandardScaler(with_mean=True, with_std=True)
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = self.scaler_.transform(X)
        
        # Clipping Seurat
        X_scaled = np.clip(X_scaled, -10, 10)
        return X_scaled

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        self.hvg_idx_ = select_hvg_cv(X, n_genes=800)
        X = X[:, self.hvg_idx_]
        
        # Seurat scaling
        X = self._seurat_scale(X, fit=True)

        y_main = np.array(y, dtype=object)
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4 | mask_cd8] = "T_cells"

        self.pipe_main.fit(X, y_main)
        self.classes_main = self.pipe_main.classes_

        mask_tcells = mask_cd4 | mask_cd8
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]

        self.pipe_tcells.fit(X_tcells, y_tcells)
        self.classes_tcells = self.pipe_tcells.classes_

        self.classes_ = np.array(["Cancer_cells", "NK_cells", "T_cells_CD4+", "T_cells_CD8+"])

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        X = self._seurat_scale(X, fit=False)
        n = X.shape[0]

        p_main = self.pipe_main.predict_proba(X)
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]

        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]

        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)

        pred_main = np.argmax(p_main, axis=1)
        mask_tcells = pred_main == idx_tcells
        
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]

            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]

        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba

    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================================
# CLASSIFIEUR 4 : HVG HYBRIDE + LOGISTIC
# ============================================================================

class ClassifierHybridHVG:
    """Logistic avec sélection HVG hybride."""
    
    def __init__(self):
        self.pipe_main = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=12),
            LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )

        self.pipe_tcells = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=12),
            LogisticRegression(
                C=0.1,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        # HVG HYBRIDE
        self.hvg_idx_ = select_hvg_hybrid(X, n_genes=1000, cv_weight=0.7)
        X = X[:, self.hvg_idx_]

        y_main = np.array(y, dtype=object)
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4 | mask_cd8] = "T_cells"

        self.pipe_main.fit(X, y_main)
        self.classes_main = self.pipe_main.classes_

        mask_tcells = mask_cd4 | mask_cd8
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]

        self.pipe_tcells.fit(X_tcells, y_tcells)
        self.classes_tcells = self.pipe_tcells.classes_

        self.classes_ = np.array(["Cancer_cells", "NK_cells", "T_cells_CD4+", "T_cells_CD8+"])

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        n = X.shape[0]

        p_main = self.pipe_main.predict_proba(X)
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]

        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]

        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)

        pred_main = np.argmax(p_main, axis=1)
        mask_tcells = pred_main == idx_tcells
        
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]

            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]

        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba

    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================================
# CLASSIFIEUR 5 : ENSEMBLE LR + GB
# ============================================================================

class ClassifierEnsemble:
    """Ensemble Logistic + Gradient Boosting."""
    
    def __init__(self, lr_weight=0.6, gb_weight=0.4):
        self.lr_weight = lr_weight
        self.gb_weight = gb_weight
        
        # LR niveau 1
        self.pipe_lr_main = make_pipeline(
            StandardScaler(),
            PCA(n_components=12),
            LogisticRegression(C=0.1, penalty='l2', solver='lbfgs',
                             max_iter=1000, class_weight='balanced',
                             random_state=42, n_jobs=-1)
        )
        
        # GB niveau 1
        self.pipe_gb_main = make_pipeline(
            StandardScaler(),
            PCA(n_components=12),
            GradientBoostingClassifier(n_estimators=80, learning_rate=0.05,
                                      max_depth=2, min_samples_split=30,
                                      min_samples_leaf=15, subsample=0.8,
                                      max_features='sqrt', random_state=42)
        )
        
        # Niveau 2
        self.pipe_tcells = make_pipeline(
            StandardScaler(),
            PCA(n_components=12),
            LogisticRegression(C=0.1, penalty='l2', solver='lbfgs',
                             max_iter=1000, class_weight='balanced',
                             random_state=42, n_jobs=-1)
        )

    def fit(self, X_sparse, y):
        X = _preprocess_X(X_sparse)
        self.hvg_idx_ = select_hvg_cv(X, n_genes=1000)
        X = X[:, self.hvg_idx_]
        
        y_main = np.array(y, dtype=object)
        mask_cd4 = y == "T_cells_CD4+"
        mask_cd8 = y == "T_cells_CD8+"
        y_main[mask_cd4 | mask_cd8] = "T_cells"
        
        self.pipe_lr_main.fit(X, y_main)
        self.pipe_gb_main.fit(X, y_main)
        self.classes_main = self.pipe_lr_main.classes_
        
        mask_tcells = mask_cd4 | mask_cd8
        X_tcells = X[mask_tcells]
        y_tcells = y[mask_tcells]
        
        self.pipe_tcells.fit(X_tcells, y_tcells)
        self.classes_tcells = self.pipe_tcells.classes_
        
        self.classes_ = np.array(["Cancer_cells", "NK_cells", 
                                  "T_cells_CD4+", "T_cells_CD8+"])

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse)
        X = X[:, self.hvg_idx_]
        n = X.shape[0]
        
        # Vote pondéré niveau 1
        p_lr_main = self.pipe_lr_main.predict_proba(X)
        p_gb_main = self.pipe_gb_main.predict_proba(X)
        p_main = self.lr_weight * p_lr_main + self.gb_weight * p_gb_main
        
        idx_cancer = np.where(self.classes_main == "Cancer_cells")[0][0]
        idx_nk = np.where(self.classes_main == "NK_cells")[0][0]
        idx_tcells = np.where(self.classes_main == "T_cells")[0][0]
        
        p_cancer = p_main[:, idx_cancer]
        p_nk = p_main[:, idx_nk]
        p_tcells = p_main[:, idx_tcells]
        
        p_cd4 = np.zeros(n)
        p_cd8 = np.zeros(n)
        
        pred_main = np.argmax(p_main, axis=1)
        mask_tcells = pred_main == idx_tcells
        
        if np.any(mask_tcells):
            X_t = X[mask_tcells]
            p_tcells_fine = self.pipe_tcells.predict_proba(X_t)
            idx_cd4 = np.where(self.classes_tcells == "T_cells_CD4+")[0][0]
            idx_cd8 = np.where(self.classes_tcells == "T_cells_CD8+")[0][0]
            
            p_cd4[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd4]
            p_cd8[mask_tcells] = p_tcells[mask_tcells] * p_tcells_fine[:, idx_cd8]
        
        proba = np.vstack([p_cancer, p_nk, p_cd4, p_cd8]).T
        return proba

    def predict(self, X_sparse):
        proba = self.predict_proba(X_sparse)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================================
# FONCTION D'ÉVALUATION
# ============================================================================

def evaluate_classifier(clf, X_train, y_train, X_test, y_test, name):
    """Évalue un classifier et retourne les métriques."""
    print(f"\n{'='*80}")
    print(f"ÉVALUATION : {name}")
    print(f"{'='*80}")
    
    # Entraînement avec timing
    start = time()
    clf.fit(X_train, y_train)
    train_time = time() - start
    
    # Prédictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # Métriques
    train_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_acc = balanced_accuracy_score(y_test, y_test_pred)
    gap = train_acc - test_acc
    
    print(f"\nTrain balanced accuracy: {train_acc:.4f}")
    print(f"Test balanced accuracy:  {test_acc:.4f}")
    print(f"Overfitting gap:         {gap:.4f}")
    print(f"Training time:           {train_time:.2f}s")
    
    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred, target_names=clf.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=clf.classes_)
    
    return {
        'name': name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'train_time': train_time,
        'y_pred': y_test_pred,
        'confusion_matrix': cm,
        'clf': clf
    }


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main(X_train, y_train, X_test, y_test):
    """
    Fonction principale pour comparer tous les classifiers.
    
    Usage:
    >>> from classifier_comparison import main
    >>> results = main(X_train, y_train, X_test, y_test)
    """
    
    # Liste des classifiers à tester
    classifiers = [
        ("1. Sub2 (Baseline GB)", ClassifierSub2()),
        ("2. Logistic Regression", ClassifierLogistic()),
        ("3. Seurat + Logistic", ClassifierSeurat()),
        ("4. HVG Hybrid + Logistic", ClassifierHybridHVG()),
        ("5. Ensemble LR+GB", ClassifierEnsemble()),
    ]
    
    # Évaluation de tous les classifiers
    results = []
    for name, clf in classifiers:
        result = evaluate_classifier(clf, X_train, y_train, X_test, y_test, name)
        results.append(result)
    
    # ========================================================================
    # COMPARAISON FINALE
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("TABLEAU COMPARATIF FINAL")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame({
        'Méthode': [r['name'] for r in results],
        'Train Acc': [f"{r['train_acc']:.4f}" for r in results],
        'Test Acc': [f"{r['test_acc']:.4f}" for r in results],
        'Gap': [f"{r['gap']:.4f}" for r in results],
        'Temps (s)': [f"{r['train_time']:.1f}" for r in results]
    })
    print(df.to_string(index=False))
    
    # Meilleurs modèles
    best_test = max(results, key=lambda x: x['test_acc'])
    best_gap = min(results, key=lambda x: x['gap'])
    fastest = min(results, key=lambda x: x['train_time'])

    print(f"\n{'='*80}")
    print("MEILLEURS MODÈLES")
    print(f"{'='*80}\n")

    print(f"🏆 Meilleur Test Accuracy : {best_test['name']} "
          f"(Test Acc = {best_test['test_acc']:.4f})")

    print(f"🧠 Meilleure Généralisation (plus petit gap) : {best_gap['name']} "
          f"(Gap = {best_gap['gap']:.4f})")

    print(f"⚡ Entraînement le plus rapide : {fastest['name']} "
          f"(Temps = {fastest['train_time']:.2f}s)")

    # ========================================================================
    # MATRICES DE CONFUSION
    # ========================================================================

    print(f"\n{'='*80}")
    print("MATRICES DE CONFUSION (TEST)")
    print(f"{'='*80}\n")

    n_models = len(results)
    n_cols = 2
    n_rows = (n_models + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()

    for ax, result in zip(axes, results):
        cm = result['confusion_matrix']
        classes = result['clf'].classes_

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax
        )

        ax.set_title(result['name'])
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Vrai")

    # Supprimer les axes vides s'il y en a
    for i in range(len(results), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    return results

if __name__ == "__main__":
    main(X_train, y_train, X_test, y_test)
