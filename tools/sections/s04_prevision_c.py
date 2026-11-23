from cellules import code, md

CELLS = [
md(r"""
### 4.8 Trois méthodes d'évaluation et leur comparaison (cours 2)
Avec les **mêmes hyperparamètres** (réglés en 4.5), on évalue chaque modèle de trois façons, sur la période entraînement + validation (2011-2021). La période de test reste intacte.
1. **Hold-out temporel** : apprentissage 2011-2019, évaluation 2020-2021 (un seul découpage).
2. **Validation croisée expanding window** : 5 plis ; la fenêtre d'apprentissage grandit à chaque pli.
3. **Validation croisée rolling (sliding) window** : mêmes blocs de validation ; la fenêtre d'apprentissage garde une taille fixe et glisse.

**Grille de lecture du cours** :
- évaluations proches → modèle **stable** ;
- erreur du hold-out plus faible que celle des CV → le découpage unique était **trop optimiste** ;
- rolling meilleur que expanding → les données anciennes dégradent la prévision : une **mémoire courte** est préférable ; l'inverse → l'historique apporte de l'information utile.
"""),
code(r"""
from sklearn.ensemble import VotingRegressor
from src.model_zoo import BASELINES
from src.temporal import oof_predictions

learned_names = [n for n in learned_val.index if "+ ACP" not in n]
three_models = {"Baseline – moyenne par actif": BASELINES["Baseline – moyenne par actif"],
                **{n: fc_fitted[n] for n in learned_names[:3] if not n.startswith("SVR")}}
if "Ridge" not in three_models:
    three_models["Ridge"] = fc_fitted["Ridge"]
rows3 = []
for name, model in three_models.items():
    t3 = evaluate_three_ways(model, tr, va, FEATS, n_splits=3 if QUICK_RUN else 5)
    t3.index = pd.MultiIndex.from_product([[name], t3.index], names=["modèle", "méthode"])
    rows3.append(t3)
three_tab = pd.concat(rows3)
three_tab.to_csv(RESULTS_DIR / "agent_prevision_trois_evaluations.csv")
RESULTS["prevision_trois_evaluations"] = three_tab
three_tab[["MSE", "R² hors échantillon", "IC moyen", "écart-type IC", "nb plis"]]
"""),
md(r"""
### 4.9 Test de Diebold-Mariano (cours 2)
Pour affirmer qu'un modèle est **meilleur** qu'un autre, il ne suffit pas d'une MSE un peu plus faible : il faut que l'écart soit **statistiquement significatif**.

- Différentiel de perte à chaque date : $d_t = L(e_{1,t}) - L(e_{2,t})$, où $L$ est l'erreur quadratique moyenne des 25 actifs à la date *t*.
- $H_0 : \mathbb{E}[d_t] = 0$ (même précision) contre $H_1 : \mathbb{E}[d_t] \neq 0$.
- $DM = \bar d / \sqrt{\widehat{Var}(\bar d)}$ avec $\widehat{Var}(\bar d) = \frac{1}{T}\left(\hat\gamma_0 + 2\sum_{k=1}^{h-1}\hat\gamma_k\right)$, où les autocovariances $\hat\gamma_k$ corrigent le chevauchement des cibles ($h$ = 5 pas). On applique la correction de petit échantillon de Harvey, Leybourne et Newbold.
- Lecture : modèle 1 = référence, modèle 2 = candidat. **DM > 0 et p < 0,05 → le candidat est meilleur** ; p > 0,05 → les deux sont statistiquement équivalents.

Deux jeux de prévisions sont testés :
- **(a) la validation 2020-2021** : 96 dates seulement, donc un test peu puissant ;
- **(b) les prévisions hors pli de la CV expanding sur 2013-2021** : environ 400 dates. Chaque date y est prédite par un modèle entraîné uniquement sur le passé, donc sans fuite, et toujours sans toucher au test.
"""),
code(r"""
losses_val = {n: loss_by_date(va["target_reg"], p) for n, p in val_preds.items()}
dm_val = dm_table(losses_val, "Baseline – prévision nulle")
print("(a) Validation 2020-2021 – référence : prévision nulle")
display(dm_val[["DM", "p-valeur", "T", "verdict"]])
"""),
code(r"""
trva = pd.concat([tr, va])
cv_oof = TemporalCV(n_splits=3 if QUICK_RUN else 5, mode="expanding")
oof = {n: oof_predictions(b, trva, FEATS, cv_oof) for n, b in BASELINES.items()}
for n in learned_names:
    oof[n] = oof_predictions(fc_fitted[n], trva, FEATS, cv_oof, max_rows=3000 if n.startswith("SVR") else None,
                             seed=RANDOM_STATE)
    print(f"✓ prévisions hors pli : {n}")
y_oof = trva.loc[oof["Baseline – prévision nulle"].index, "target_reg"]
oof_mse = pd.Series({n: float(((y_oof - p.reindex(y_oof.index)) ** 2).mean()) for n, p in oof.items()})

# Ensemble : moyenne des prévisions des 3 meilleurs modèles appris (MSE hors pli)
top3_oof = oof_mse.drop([n for n in oof_mse.index if n.startswith("Baseline")]).sort_values().index[:3].tolist()
oof["Ensemble (moyenne des 3 meilleurs)"] = sum(oof[n] for n in top3_oof) / 3
print("Ensemble construit avec :", top3_oof)
"""),
code(r"""
trva_oof = trva.loc[y_oof.index]
oof_report = pd.DataFrame({n: forecast_report(p.reindex(y_oof.index), trva_oof) for n, p in oof.items()}).T.sort_values("MSE")
losses_oof = {n: loss_by_date(y_oof, p.reindex(y_oof.index)) for n, p in oof.items()}
dm_null = dm_table(losses_oof, "Baseline – prévision nulle")
dm_asset = dm_table(losses_oof, "Baseline – moyenne par actif")
dm_oof = pd.DataFrame({"MSE hors pli": oof_report["MSE"], "IC moyen": oof_report["IC moyen"],
                       "IC t-stat (NW)": oof_report["IC t-stat (Newey-West)"],
                       "DM vs prévision nulle": dm_null["DM"], "p vs nulle": dm_null["p-valeur"],
                       "DM vs moyenne par actif": dm_asset["DM"], "p vs moyenne par actif": dm_asset["p-valeur"]})
dm_oof = dm_oof.sort_values("MSE hors pli")
dm_oof.to_csv(RESULTS_DIR / "agent_prevision_diebold_mariano.csv")
RESULTS["prevision_diebold_mariano"] = dm_oof
print(f"(b) Prévisions hors pli : {y_oof.index.get_level_values('date').nunique()} dates, "
      f"{y_oof.index.get_level_values('date').min().date()} → {y_oof.index.get_level_values('date').max().date()}")
dm_oof.style.format("{:.4f}").background_gradient(subset=["p vs nulle"], cmap="RdYlGn_r", vmin=0, vmax=0.2)
"""),
md(r"""
### 4.10 Choix du modèle (règle fixée avant de voir les résultats)
1. **Critère** : la MSE hors pli sur 2013-2021, plus fiable que la seule validation 2020-2021 (quatre fois plus de dates).
2. **Garde-fou** : l'agent Prévision n'influence l'allocation que si le modèle retenu bat la **prévision nulle** au test de Diebold-Mariano (DM > 0, p < 0,05). Sinon, son poids est mis à zéro ($\alpha = 0$) et l'allocation repose uniquement sur le risque et l'allocation stratégique du profil. Le modèle reste affiché à titre d'information.
3. Si le modèle est activé, son signal entre dans l'allocation par la loi fondamentale de la gestion active (Grinold & Kahn) : $\alpha_i = IC \times \sigma_i \times z_i$, avec $z_i$ la prévision standardisée entre actifs, $\sigma_i$ la volatilité de l'actif, et un IC divisé par deux par prudence (l'IC mesuré surestime en général la performance future).
"""),
code(r"""
cands = dm_oof.drop(index=[i for i in dm_oof.index if i.startswith("Baseline")])
best_fc_name = cands["MSE hors pli"].idxmin()
dm_best = cands.loc[best_fc_name]
FC_ACTIVE = bool(dm_best["DM vs prévision nulle"] > 0 and dm_best["p vs nulle"] < 0.05)
IC_VAL = float(dm_best["IC moyen"])
IC_USED = IC_SHRINK * IC_VAL if (FC_ACTIVE and IC_VAL > 0) else 0.0
print(f"Modèle retenu : {best_fc_name}")
print(f"  MSE hors pli = {dm_best['MSE hors pli']:.5f} | IC = {IC_VAL:.4f} (t = {dm_best['IC t-stat (NW)']:.2f})")
print(f"  DM vs prévision nulle : {dm_best['DM vs prévision nulle']:.2f} (p = {dm_best['p vs nulle']:.3f})")
print(f"  DM vs moyenne par actif : {dm_best['DM vs moyenne par actif']:.2f} (p = {dm_best['p vs moyenne par actif']:.3f})")
print("→ Agent Prévision", "ACTIVÉ" if FC_ACTIVE else "NON ACTIVÉ (α = 0)", f"| IC utilisé = {IC_USED:.4f}")


def build_forecast_model(name):
    # Pipeline du modèle retenu (un ensemble est un VotingRegressor des 3 meilleurs pipelines)
    if name.startswith("Ensemble"):
        return VotingRegressor([(f"m{i}", clone(fc_fitted[n])) for i, n in enumerate(top3_oof)])
    return clone(fc_fitted[name])


chrono("4c. Prévision : 3 évaluations, Diebold-Mariano, choix")
"""),
]
