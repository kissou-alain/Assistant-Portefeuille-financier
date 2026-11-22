"""
Analyse en Composantes Principales (cours 2, « Techniques de réduction de dimensions »).

Étapes du cours (diapo « Résumé ») :
1. standardiser les données X ;
2. matrice de variance-covariance S = XᵀX / (n − 1) (= matrice de corrélation, car X est réduite) ;
3. valeurs propres λ_k et vecteurs propres V_k de S, triés par ordre décroissant ;
4. projection des individus : F = X·P (facteurs ou composantes principales) ;
5. choix du nombre de facteurs : Kaiser (λ ≥ 1), variance cumulée 80-90 %, coude ;
6. corrélations variables–facteurs pour interpréter les axes.

Vérification préalable (N.B. du cours) : l'ACP n'a d'intérêt que si les variables sont
corrélées → indice KMO ≥ 0,6 et test de sphéricité de Bartlett significatif.
Toutes les statistiques sont calculées sur l'ENTRAÎNEMENT uniquement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def kmo(X: pd.DataFrame) -> tuple[float, pd.Series]:
    """Indice de Kaiser-Meyer-Olkin.

    KMO = Σ r_ij² / (Σ r_ij² + Σ a_ij²)   (i ≠ j)
    où r_ij est la corrélation simple et a_ij la corrélation partielle (obtenue à partir de
    l'inverse de la matrice de corrélation). Une valeur proche de 1 indique que les
    corrélations entre variables s'expliquent par des facteurs communs : l'ACP est utile.
    Retourne (KMO global, KMO par variable).
    """
    R = np.corrcoef(np.asarray(X, float), rowvar=False)
    Rinv = np.linalg.pinv(R)
    d = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    A = -Rinv / d                        # corrélations partielles
    np.fill_diagonal(R, 0)
    np.fill_diagonal(A, 0)
    r2, a2 = R ** 2, A ** 2
    per_var = r2.sum(0) / (r2.sum(0) + a2.sum(0))
    return float(r2.sum() / (r2.sum() + a2.sum())), pd.Series(per_var, index=X.columns)


def bartlett_sphericity(X: pd.DataFrame) -> dict:
    """Test de sphéricité de Bartlett.

    H0 : la matrice de corrélation est l'identité (variables non corrélées, ACP inutile).
    χ² = −(n − 1 − (2p + 5)/6) · ln|R|,   ddl = p(p − 1)/2
    """
    X = np.asarray(X, float)
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    sign, logdet = np.linalg.slogdet(R)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    ddl = p * (p - 1) / 2
    return {"chi2": float(chi2), "ddl": int(ddl), "p-valeur": float(stats.chi2.sf(chi2, ddl))}


class ACP:
    """ACP « à la main » selon le cours (diagonalisation de S), pour l'analyse et les graphiques.
    Dans les pipelines de prévision on utilise sklearn.decomposition.PCA (même résultat)."""

    def fit(self, X: pd.DataFrame):
        self.columns_ = list(X.columns)
        self.mean_ = X.mean()
        self.std_ = X.std(ddof=1)
        Z = ((X - self.mean_) / self.std_).values
        S = Z.T @ Z / (len(Z) - 1)
        vals, vecs = np.linalg.eigh(S)                     # S symétrique : vecteurs propres orthogonaux
        order = np.argsort(vals)[::-1]
        self.eigenvalues_ = vals[order]
        self.P_ = vecs[:, order]
        # convention de signe : la plus grande composante de chaque axe est positive
        self.P_ *= np.sign(self.P_[np.abs(self.P_).argmax(0), range(self.P_.shape[1])])
        return self

    def transform(self, X: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
        Z = ((X[self.columns_] - self.mean_) / self.std_).values
        F = Z @ self.P_[:, :k]
        return pd.DataFrame(F, index=X.index, columns=[f"F{j + 1}" for j in range(F.shape[1])])

    def variance_table(self) -> pd.DataFrame:
        """Valeur propre, % de variance expliquée λ_k / Σλ_j et % cumulé."""
        lam = self.eigenvalues_
        return pd.DataFrame({"valeur propre": lam, "% variance": lam / lam.sum() * 100,
                             "% cumulé": np.cumsum(lam) / lam.sum() * 100},
                            index=[f"F{j + 1}" for j in range(len(lam))])

    def n_components(self) -> dict:
        """Nombre de facteurs proposé par chaque critère du cours."""
        lam = self.eigenvalues_
        cum = np.cumsum(lam) / lam.sum()
        # coude : point où la baisse des valeurs propres ralentit le plus (dérivée seconde maximale)
        coude = int(np.argmax(np.diff(lam, 2)) + 2) if len(lam) > 3 else len(lam)
        return {"Kaiser (λ ≥ 1)": int((lam >= 1).sum()),
                "80 % de variance": int(np.searchsorted(cum, 0.80) + 1),
                "90 % de variance": int(np.searchsorted(cum, 0.90) + 1),
                "Coude": coude}

    def correlations(self, k: int = 3) -> pd.DataFrame:
        """Corrélation variable–facteur : cor(X_j, F_k) = v_jk · √λ_k (variables réduites)."""
        C = self.P_[:, :k] * np.sqrt(self.eigenvalues_[:k])
        return pd.DataFrame(C, index=self.columns_, columns=[f"F{j + 1}" for j in range(k)])
