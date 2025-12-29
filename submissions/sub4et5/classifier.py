import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def _preprocess_X(X_sparse):
    X = X_sparse.toarray().astype(np.float32)
    counts = X.sum(axis=1)[:, None]
    counts[counts == 0] = 1
    X = X / counts * 1e4
    return np.log1p(X)


def select_hvg_improved(X, n_genes=1000): #800 pour sub5
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    
    cv = np.divide(variances, means, 
                   out=np.zeros_like(variances), 
                   where=means > 0.1)
    
    idx = np.argsort(cv)[-n_genes:]
    return idx


class Classifier:
    
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
        self.hvg_idx_ = select_hvg_improved(X, n_genes=1000) #800 pour sub5
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
        y_pred_idx = np.argmax(proba, axis=1)
        return self.classes_[y_pred_idx]
    
