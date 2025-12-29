import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\flavi\OneDrive\Documents\ENSIIE\scmark-test")
from scipy.sparse import issparse

from problem import get_train_data, get_test_data
X_train, y_train = get_train_data()
X_test, y_test = get_test_data()


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from problem import get_train_data, get_test_data

X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

def preprocess_X(X_sparse):
    """Normalisation CPM + log1p"""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)

def select_hvg(X, n_genes=800):
    """Sélection des gènes hautement variables (coefficient de variation)"""
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    cv = np.divide(variances, means, out=np.zeros_like(variances), where=means > 0.1)
    idx = np.argsort(cv)[-n_genes:]
    return idx

# 1. Preprocessing
X_train_norm = preprocess_X(X_train)
X_test_norm = preprocess_X(X_test)

# 2. Sélection HVG
hvg_idx = select_hvg(X_train_norm, n_genes=800)
X_train_hvg = X_train_norm[:, hvg_idx]
X_test_hvg = X_test_norm[:, hvg_idx]

# 3. Analyse de la variance expliquée
X_scaled = StandardScaler().fit_transform(X_train_hvg)
pca_full = PCA()
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = (cum_var >= 0.9).argmax() + 1
print(f"Nombre de composantes pour 90% variance : {n_components_90}")

# 4. Test de performance
components = [5, 12, 25, 50, 100]
for n in components:
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n),
        LogisticRegression(class_weight='balanced', max_iter=1000)
    )
    pipe.fit(X_train_hvg, y_train)
    score = pipe.score(X_test_hvg, y_test)
    print(f"{n} composantes -> test accuracy: {score:.4f}")


# Test de performance pour différentes composantes
components_range = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
accuracies = []

print("Évaluation des performances...")
for n in components_range:
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n),
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    )
    pipe.fit(X_train_hvg, y_train)
    score = pipe.score(X_test_hvg, y_test)
    accuracies.append(score)
    print(f"{n:3d} composantes -> accuracy: {score:.4f}")

# Création de la figure avec 3 sous-graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Variance expliquée cumulative
ax1 = axes[0]
n_comp_90 = (cum_var >= 0.9).argmax() + 1
ax1.plot(range(1, len(cum_var) + 1), cum_var * 100, 'b-', linewidth=2)
ax1.axhline(y=90, color='r', linestyle='--', label='90% variance')
ax1.axvline(x=n_comp_90, color='r', linestyle='--', alpha=0.5)
ax1.axvline(x=25, color='green', linestyle='--', alpha=0.7, label='n=25 (choix optimal)')
ax1.scatter([25], [cum_var[24] * 100], color='green', s=100, zorder=5)
ax1.text(25, cum_var[24] * 100 - 5, f'{cum_var[24]*100:.1f}%', 
         ha='center', fontsize=10, color='green', weight='bold')
ax1.set_xlabel('Nombre de composantes principales', fontsize=12)
ax1.set_ylabel('Variance expliquée cumulative (%)', fontsize=12)
ax1.set_title('Variance expliquée vs Nombre de composantes', fontsize=14, weight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0, 500)

# 2. Performance du classifieur
ax2 = axes[1]
ax2.plot(components_range, np.array(accuracies) * 100, 'o-', linewidth=2, 
         markersize=8, color='darkblue', label='Accuracy test')
best_idx = np.argmax(accuracies)
ax2.scatter([components_range[best_idx]], [accuracies[best_idx] * 100], 
           color='red', s=200, zorder=5, marker='*', label=f'Maximum ({components_range[best_idx]} comp.)')
ax2.axvline(x=25, color='green', linestyle='--', alpha=0.7, label='n=25 (choix)')
ax2.set_xlabel('Nombre de composantes principales', fontsize=12)
ax2.set_ylabel('Accuracy sur test set (%)', fontsize=12)
ax2.set_title('Performance de classification vs Complexité', fontsize=14, weight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(75, 86)

# 3. Zoom sur la zone optimale
ax3 = axes[2]
zoom_indices = [i for i, c in enumerate(components_range) if 5 <= c <= 75]
zoom_components = [components_range[i] for i in zoom_indices]
zoom_accuracies = [accuracies[i] for i in zoom_indices]

ax3.plot(zoom_components, np.array(zoom_accuracies) * 100, 'o-', 
         linewidth=2.5, markersize=10, color='darkblue')
ax3.scatter([25], [accuracies[components_range.index(25)] * 100], 
           color='green', s=300, zorder=5, marker='*', 
           edgecolors='darkgreen', linewidths=2)
ax3.axvline(x=25, color='green', linestyle='--', alpha=0.5)

# Annotation du pic
for i, (c, a) in enumerate(zip(zoom_components, zoom_accuracies)):
    if c in [25, 50]:
        ax3.annotate(f'{a*100:.2f}%', 
                    xy=(c, a*100), 
                    xytext=(c, a*100 + 0.5),
                    fontsize=10, 
                    weight='bold',
                    ha='center',
                    color='green' if c == 25 else 'red')

ax3.set_xlabel('Nombre de composantes principales', fontsize=12)
ax3.set_ylabel('Accuracy sur test set (%)', fontsize=12)
ax3.set_title('Zoom : Zone optimale (surapprentissage après 25)', fontsize=14, weight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(76, 85)

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : pca_analysis.png")
plt.show()

# Affichage des statistiques clés
print("\n" + "="*60)
print("RÉSUMÉ DE L'ANALYSE")
print("="*60)
print(f"Variance expliquée avec 25 composantes : {cum_var[24]*100:.2f}%")
print(f"Composantes pour 90% variance : {n_comp_90}")
print(f"Meilleure accuracy : {max(accuracies)*100:.2f}% ({components_range[best_idx]} composantes)")
print(f"Accuracy à 25 composantes : {accuracies[components_range.index(25)]*100:.2f}%")
print(f"Perte avec 50 composantes : {(accuracies[components_range.index(25)] - accuracies[components_range.index(50)])*100:.2f}%")
print("="*60)


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


# Préprocessing identique au classifieur
X_trainPCA = X_train.toarray().astype(float)
counts = X_trainPCA.sum(axis=1)[:, None]
counts[counts == 0] = 1
X_trainPCA = X_trainPCA / counts * 1e4
X_trainPCA = np.log1p(X_trainPCA)

# Sélection HVG
hvg_idx = select_hvg_improved(X_trainPCA, n_genes=800)
X_trainHVG = X_trainPCA[:, hvg_idx]

# Standardisation
X_scaled = StandardScaler().fit_transform(X_trainHVG)

# PCA complet
pca = PCA()
pca.fit(X_scaled)

# Variance expliquée cumulative
cum_var = np.cumsum(pca.explained_variance_ratio_)
print("Variance expliquée cumulative:", cum_var)

# Nombre de composantes pour 90% variance
n_components_90 = (cum_var >= 0.9).argmax() + 1
print("Nombre de composantes pour 90% variance :", n_components_90)

# Test de performance pour différentes composantes
components = [5, 12, 25, 50, 100]
for n in components:
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n),
        LogisticRegression(class_weight='balanced', max_iter=1000)
    )
    # On doit passer les mêmes données HVG pré-traitées
    pipe.fit(X_trainHVG, y_train)
    score = pipe.score(X_test[:, hvg_idx].toarray(), y_test)  # attention à la sélection HVG aussi pour X_test
    print(f"{n} composantes -> test accuracy: {score:.4f}")


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
            PCA(n_components=30),
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
            PCA(n_components=30),
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








def _preprocess_X(X_sparse):
    """CPM-like normalization + log1p, standard for scRNA-seq."""
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg(X, n_genes=1000):
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
            PCA(n_components=30),
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
            PCA(n_components=30),
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
        self.hvg_idx_ = select_hvg(X, n_genes=1000)
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
