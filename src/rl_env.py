"""
Agent d'apprentissage par renforcement pour l'allocation (cours 1, « Apprentissage par
renforcement – exemple en gestion de portefeuille »).

| Composante     | Dans ce projet                                                             |
|----------------|----------------------------------------------------------------------------|
| Agent          | le gestionnaire de portefeuille automatisé (réseau de neurones, PPO)        |
| Environnement  | le marché, rejoué mois par mois à partir des données historiques           |
| État s_t       | instantané du marché : variables de régime, probabilités γ du GMM,          |
|                | rendements et volatilités récents de chaque classe, poids actuels          |
| Action a_t     | vecteur d'allocation entre 5 classes d'actifs (somme = 1)                   |
| Récompense r_t | rendement du mois − pénalité de risque − coûts de transaction              |
| Politique π    | la stratégie apprise π(s) → a                                              |

Récompense (utilité moyenne-variance du profil) :
    r_t = Σ_j R_j − (λ/2) Σ_j R_j² − c · Σ_i |w_i − w_i^préc|
où les R_j sont les rendements journaliers du portefeuille pendant le mois, λ l'aversion au
risque du profil et c le coût de transaction. L'agent maximise la récompense cumulée actualisée
G_t = Σ_k γ^k r_{t+k}.

Les contraintes du profil (plafond actions, part défensive minimale, plafond monétaire,
plafond matières premières) sont imposées en PROJETANT l'action sur l'ensemble des poids admissibles.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from scipy.optimize import minimize

RL_CLASSES = ["Actions", "Immobilier", "Obligations", "Monétaire", "Matières premières"]
STEP = 21                      # un pas de temps = un mois de bourse
COST = 10 / 1e4                # 10 points de base par unité échangée


def class_returns(returns: pd.DataFrame, asset_class: dict) -> pd.DataFrame:
    """Rendement journalier de chaque classe = moyenne des actifs disponibles de la classe."""
    out = {}
    for c in RL_CLASSES:
        members = [t for t in returns.columns if asset_class.get(t) == c]
        out[c] = returns[members].mean(axis=1, skipna=True)
    return pd.DataFrame(out).fillna(0.0)


def project_weights(w0: np.ndarray, cons: dict) -> np.ndarray:
    """Projection euclidienne de w0 sur les poids admissibles du profil :
    min ||w − w0||²  s.c.  Σw = 1, w ≥ 0, actions + immobilier ≤ equity_max,
    obligations + monétaire ≥ defensive_min, monétaire ≤ cash_max, matières premières ≤ commodity_max."""
    eq, bond, cash, com = 0, 2, 3, 4
    c = [{"type": "eq", "fun": lambda w: w.sum() - 1},
         {"type": "ineq", "fun": lambda w: cons["equity_max"] - w[eq] - w[1]},
         {"type": "ineq", "fun": lambda w: w[bond] + w[cash] - cons["defensive_min"]},
         {"type": "ineq", "fun": lambda w: cons["cash_max"] - w[cash]},
         {"type": "ineq", "fun": lambda w: cons["commodity_max"] - w[com]}]
    res = minimize(lambda w: ((w - w0) ** 2).sum(), w0, jac=lambda w: 2 * (w - w0),
                   bounds=[(0, 1)] * len(w0), constraints=c, method="SLSQP", options={"maxiter": 100})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


class PortfolioEnv(gym.Env):
    """Environnement Gymnasium : un épisode = `episode_months` mois tirés au hasard dans [start, end]
    (entraînement), ou toute la période d'un seul tenant (évaluation, random_start=False)."""

    metadata = {"render_modes": []}

    def __init__(self, cls_ret: pd.DataFrame, state_feats: pd.DataFrame, cons: dict, risk_aversion: float,
                 start, end, episode_months: int = 24, random_start: bool = True, seed: int = 0):
        super().__init__()
        idx = cls_ret.loc[start:end].index
        self.dates = idx
        self.R = cls_ret.loc[idx].values
        # variables d'état propres aux classes : rendements 1 et 3 mois, volatilité 3 mois (annualisée)
        ret21 = cls_ret.rolling(21).sum().loc[idx].values
        ret63 = cls_ret.rolling(63).sum().loc[idx].values
        vol63 = (cls_ret.rolling(63).std() * np.sqrt(252)).loc[idx].values
        self.X = np.hstack([state_feats.reindex(idx).ffill().fillna(0).values, ret21, ret63, vol63])
        self.cons, self.lam = cons, risk_aversion
        self.episode_months, self.random_start = episode_months, random_start
        self.n = len(RL_CLASSES)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.X.shape[1] + self.n,), dtype=np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, shape=(self.n,), dtype=np.float32)   # logits
        self._seed, self._seeded = seed, False

    def _obs(self):
        return np.concatenate([self.X[self.t], self.w]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is None and not self._seeded:        # reproductibilité : graine fixée au premier épisode
            seed = self._seed
        self._seeded = True
        super().reset(seed=seed)
        max_start = len(self.R) - STEP * self.episode_months - 1
        self.t = int(self.np_random.integers(0, max(max_start, 1))) if self.random_start else 0
        self.t_end = self.t + STEP * self.episode_months if self.random_start else len(self.R) - 1
        self.w = np.full(self.n, 1 / self.n)
        return self._obs(), {}

    def action_to_weights(self, action):
        z = np.asarray(action, dtype=float)
        w = np.exp(z - z.max())
        return project_weights(w / w.sum(), self.cons)          # softmax puis projection

    def step(self, action):
        w_new = self.action_to_weights(action)
        cost = COST * np.abs(w_new - self.w).sum()
        end = min(self.t + STEP, self.t_end)
        w, daily = w_new.copy(), []
        for j in range(self.t + 1, end + 1):                      # les poids dérivent avec les prix
            r = self.R[j]
            rp = float(w @ r)
            daily.append(rp)
            w = w * (1 + r) / (1 + rp)
        daily = np.array(daily)
        daily[0] -= cost if len(daily) else 0
        reward = float(daily.sum() - 0.5 * self.lam * (daily ** 2).sum()) * 100   # ×100 : échelle adaptée à PPO
        self.last_daily, self.last_weights = daily, w_new
        self.w, self.t = w, end
        done = self.t >= self.t_end
        return self._obs(), reward, done, False, {}


def rollout(model, env: PortfolioEnv) -> tuple[pd.Series, pd.DataFrame]:
    """Applique la politique apprise (de façon déterministe) sur toute la période de l'environnement.
    Retourne les rendements journaliers nets et les poids choisis chaque mois."""
    obs, _ = env.reset()
    rets, weights, done = [], {}, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        d0 = env.dates[env.t]
        obs, _, done, _, _ = env.step(action)
        weights[d0] = env.last_weights
        rets.extend(zip(env.dates[env.t - len(env.last_daily) + 1: env.t + 1], env.last_daily))
    r = pd.Series(dict(rets)).sort_index()
    return r, pd.DataFrame(weights, index=RL_CLASSES).T
