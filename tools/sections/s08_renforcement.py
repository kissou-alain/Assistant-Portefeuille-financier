from cellules import code, md

CELLS = [
md(r"""
## 8. Agent d'apprentissage par renforcement (PPO)

### 8.1 Formulation (cours 1, « exemple en gestion de portefeuille »)
Un **agent** interagit avec un **environnement** : à chaque pas de temps il observe un état $s_t$, choisit une action $a_t$ selon sa politique $\pi(a\mid s)$, reçoit une récompense $r_t$ et un nouvel état $s_{t+1}$. Il cherche la politique qui maximise la **récompense cumulée actualisée** $G_t = \sum_{k\ge0}\gamma^k r_{t+k}$, où $\gamma\in[0,1[$ est le facteur d'actualisation.

| Composante | Dans ce projet |
|---|---|
| Agent | le gestionnaire de portefeuille automatisé (réseau de neurones 2 × 64) |
| Environnement | le marché historique, rejoué **mois par mois** (environnement `Gymnasium` personnalisé, `src/rl_env.py`) |
| État $s_t$ | 5 variables de régime + 3 probabilités $\gamma_{ik}$ du GMM + rendements à 1 et 3 mois et volatilité à 3 mois de chaque classe + poids actuels (25 nombres) |
| Action $a_t$ | allocation entre **5 classes d'actifs** (actions, immobilier, obligations, monétaire, matières premières), réparties à parts égales dans chaque classe |
| Récompense $r_t$ | $\sum_j R_j - \frac{\lambda}{2}\sum_j R_j^2 - c\,\lVert w_t - w_{t-1}\rVert_1$ : rendement du mois − pénalité de risque (aversion λ du profil) − coûts de transaction |
| Politique $\pi$ | apprise par **PPO** (*Proximal Policy Optimization*, bibliothèque `stable-baselines3`) |

**PPO** appartient aux méthodes **fondées sur la politique** (cours 1) : un réseau de neurones de poids θ reçoit l'état et renvoie directement une distribution sur les actions $\pi(a\mid s;\theta)$. Le dilemme **exploration / exploitation** est géré par le caractère aléatoire de cette distribution pendant l'entraînement (l'agent essaie des allocations variées), puis la politique est utilisée de façon déterministe en évaluation.

**Contraintes du profil** : l'action brute (5 nombres) passe par un softmax (poids positifs de somme 1), puis est **projetée** sur l'ensemble des poids admissibles du profil (plafond actions, part défensive minimale, plafonds monétaire et matières premières). L'agent ne peut donc jamais proposer un portefeuille interdit.

**Protocole honnête** : entraînement sur 2011-2019 (épisodes de 24 mois tirés au hasard), choix entre plusieurs graines aléatoires sur la **validation 2020-2021**, puis **une seule évaluation** sur le test, comparée aux stratégies de la section 7.

**Limites assumées** : environ 108 mois d'entraînement seulement, c'est très peu pour l'apprentissage par renforcement ; le Bitcoin est exclu de l'environnement (pas d'historique avant 2014) ; le **budget de volatilité** du profil n'est pas une contrainte stricte comme dans l'agent Allocation, il est seulement **pénalisé** dans la récompense (terme $\frac{\lambda}{2}\sum_j R_j^2$) : la volatilité réalisée peut donc dépasser la cible.
"""),
code(r"""
from stable_baselines3 import PPO
from src.rl_env import class_returns, PortfolioEnv, rollout, RL_CLASSES

cls_ret = class_returns(returns, ASSET_CLASS)
Z_state = pd.DataFrame(reg_scaler.transform(mkt_feat[reg_cols].ffill().fillna(M_tr.median())),
                       index=mkt_feat.index, columns=reg_cols)
state_feats = pd.concat([Z_state, proba_all], axis=1)
RL_STEPS = 3_000 if QUICK_RUN else 30_000
RL_SEEDS = [0] if QUICK_RUN else [0, 1]
rl_start, rl_val, rl_test = "2011-03-01", ("2020-01-01", VAL_END), (reb[0], last_date)


def make_env(prof, start, end, random_start=True, seed=0):
    c = PROFILE_CONSTRAINTS[prof]
    return PortfolioEnv(cls_ret, state_feats, c, c["risk_aversion"], start, end, random_start=random_start, seed=seed)


rl_models, rl_val_rows = {}, {}
for prof in PROFILE_CLASSES:
    for seed in RL_SEEDS:
        t0 = time.time()
        model = PPO("MlpPolicy", make_env(prof, rl_start, TRAIN_END, seed=seed), gamma=0.9, n_steps=256, batch_size=64,
                    learning_rate=3e-4, ent_coef=0.0, seed=seed, verbose=0,
                    policy_kwargs={"net_arch": [64, 64]}).learn(RL_STEPS)
        r_val, _ = rollout(model, make_env(prof, *rl_val, random_start=False))
        m = perf_metrics(r_val, rf_daily)
        rl_val_rows[(prof, seed)] = {"Sharpe validation": m["Sharpe"], "Rendement annualisé": m["Rendement annualisé"],
                                     "Volatilité": m["Volatilité annualisée"], "durée (s)": time.time() - t0}
        rl_models[(prof, seed)] = model
        print(f"✓ PPO {prof:<10} graine {seed} : Sharpe validation = {m['Sharpe']:.2f} ({time.time() - t0:.0f} s)")
rl_val_tab = pd.DataFrame(rl_val_rows).T
rl_val_tab.index.names = ["profil", "graine"]
best_seed = rl_val_tab["Sharpe validation"].groupby(level="profil").idxmax()
display(rl_val_tab)
"""),
md(r"""
### 8.2 Évaluation unique sur le test et comparaison
"""),
code(r"""
rl_rows, rl_returns, rl_weights = {}, {}, {}
for prof in PROFILE_CLASSES:
    model = rl_models[best_seed[prof]]
    r_te, w_te = rollout(model, make_env(prof, *rl_test, random_start=False))
    turnover = w_te.diff().abs().sum(1).sum() / (len(r_te) / 252)
    rl_returns[f"PPO – {prof}"], rl_weights[prof] = r_te, w_te
    model.save(MODELS_DIR / f"agent_rl_ppo_{prof}.zip")
for name, r in rl_returns.items():
    prof = name.split(" – ")[1]
    rl_rows[name] = perf_metrics(r, rf_daily, rl_weights[prof].diff().abs().sum(1).sum() / (len(r) / 252))
comp_rl = pd.concat([pd.DataFrame(rl_rows).T,
                     bt_table.loc[[f"{s} – {p}" for p in PROFILE_CLASSES
                                   for s in ("Multi-agent", "Référence stratégique", "1/N vol. cible")]]])
comp_rl = comp_rl.loc[[f"{s} – {p}" for p in PROFILE_CLASSES
                       for s in ("PPO", "Multi-agent", "Référence stratégique", "1/N vol. cible")]]
comp_rl.to_csv(RESULTS_DIR / "agent_rl_comparaison_test.csv")
RESULTS["rl_test"] = comp_rl
display(comp_rl[["Rendement annualisé", "Volatilité annualisée", "Sharpe", "Max drawdown", "Rotation annuelle"]]
        .style.format({"Sharpe": "{:.2f}", "Rotation annuelle": "{:.2f}", "Rendement annualisé": "{:.2%}",
                       "Volatilité annualisée": "{:.2%}", "Max drawdown": "{:.2%}"}))

sig_rl = {}
for p in PROFILE_CLASSES:
    for ref in ("Multi-agent", "Référence stratégique", "1/N vol. cible"):
        sig_rl[f"{p} : PPO vs {ref}"] = block_bootstrap_sharpe_diff(rl_returns[f"PPO – {p}"], bt_returns[f"{ref} – {p}"])
sig_rl = pd.DataFrame(sig_rl).T
sig_rl["significatif (5 %)"] = np.where((sig_rl["IC 95 % bas"] > 0) | (sig_rl["IC 95 % haut"] < 0), "oui", "non")
sig_rl.to_csv(RESULTS_DIR / "agent_rl_significativite.csv")
RESULTS["rl_significativite"] = sig_rl
display(sig_rl)
"""),
code(r"""
fig, axes = plt.subplots(2, 3, figsize=(19, 9))
for j, prof in enumerate(PROFILE_CLASSES):
    ax = axes[0, j]
    w_rl = (1 + rl_returns[f"PPO – {prof}"]).cumprod()
    w_rl.plot(ax=ax, lw=2, label="PPO (renforcement)")
    for s, style in (("Multi-agent", "-"), ("Référence stratégique", ":"), ("1/N vol. cible", "--")):
        (1 + bt_returns[f"{s} – {prof}"].reindex(w_rl.index).fillna(0)).cumprod().plot(ax=ax, ls=style, lw=1.3, label=s)
    ax.set_title(f"Profil {prof} – test")
    ax.legend(fontsize=8)
    rl_weights[prof].plot.area(ax=axes[1, j], linewidth=0, legend=(j == 0))
    axes[1, j].set_ylim(0, 1)
    axes[1, j].set_title(f"Allocation choisie par PPO – {prof}")
plt.tight_layout()
savefig("20_rl_ppo")
plt.show()
chrono("8. Apprentissage par renforcement (PPO)")
"""),
md(r"""
**Règle de décision (fixée à l'avance)** : l'agent PPO ne remplace l'optimiseur classique dans l'application que s'il le bat de façon **significative** (intervalle de confiance du bootstrap entièrement au-dessus de 0) pour un profil. Dans le cas contraire, il reste une **démonstration** de la méthode, et l'application garde l'agent Allocation fondé sur l'optimisation moyenne-variance, plus stable et plus facile à expliquer.
"""),
]
