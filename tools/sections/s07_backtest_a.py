from cellules import code, md

CELLS = [
md(r"""
## 7. Backtest walk-forward sur la période de test (2022 → fin)

### 7.1 Protocole (aucune information future)
- **Rebalancement mensuel** (dernier jour de bourse du mois) sur toute la période de test.
- **Covariance** estimée sur les 252 derniers jours ; **coûts de transaction** de 10 points de base par unité échangée.
- Si l'agent Prévision est activé, son modèle est **ré-entraîné chaque 1er janvier** avec les seules cibles connues à cette date (fenêtre croissante + embargo).
- Deux réglages ne sont pas appris par un modèle : la **pénalité de rotation γ** et l'**intensité de la réduction du risque en régime de stress**. Ils sont choisis sur la **validation 2020-2021**, jamais sur le test.

**Stratégies comparées**

| Stratégie | Type | Niveau de risque |
|---|---|---|
| **Multi-agent** (par profil) | Système proposé | cible du profil |
| Référence stratégique (par profil) | Allocation de référence du profil, rebalancée chaque mois, sans optimisation ni régime | proche de la cible |
| 1/N à volatilité cible (par profil) | Poids égaux mélangés avec la poche monétaire pour ne pas dépasser la cible | ≤ cible |
| Parité de risque à volatilité cible (par profil) | Contributions au risque égales, puis même mélange | ≤ cible |
| HRP à volatilité cible (par profil) | Parité de risque hiérarchique, puis même mélange | ≤ cible |
| 1/N, 60/40, variance minimale, Markowitz historique | Références classiques (un seul portefeuille) | libre |

Les références « à volatilité cible » permettent une comparaison **à risque comparable** : comparer le profil Prudent (cible 6 %) au 1/N brut (≈ 12 % de volatilité) ne serait pas équitable. **Limite** : sans emprunt (pas de levier), on peut seulement *réduire* le risque d'un portefeuille en ajoutant du monétaire, pas l'augmenter. La parité de risque et HRP, naturellement peu risquées (≈ 5 à 7 %), restent donc **en dessous** des cibles des profils Équilibré (10 %) et Dynamique (16 %). Pour ces profils, la comparaison porte sur le **ratio de Sharpe** (rendement par unité de risque) plutôt que sur le rendement brut.
"""),
code(r"""
from src.backtest import (perf_metrics, simulate, equal_weight, sixty_forty, min_variance, markowitz_historical,
                          rebalance_dates, vol_targeted_equal_weight)

emb = pd.Timedelta(days=EMBARGO_DAYS)
panel_dates = set(panel.index.get_level_values("date"))


def forecast_model_trained_until(cutoff):
    # Modèle de prévision entraîné avec les seules cibles connues à la date `cutoff`
    d_all = panel_s.index.get_level_values("date")
    data = panel_s[d_all <= cutoff - emb]
    return build_forecast_model(best_fc_name).fit(data[FEATS], data["target_reg"])


def run_multi_agent(dates, prof, reg_bundle, fc_bundle):
    orch = PortfolioOrchestrator(None, reg_bundle, fc_bundle)
    W_, prev_ = {}, None
    for d in dates:
        r = orch.recommend(d, prices, panel, mkt_feat, profile=prof, prev_w=prev_, explain=False)
        W_[d], prev_ = r["weights"], r["weights"]
    return W_
"""),
md(r"""
### 7.2 Choix des réglages sur la validation (2020-2021)
Grille : γ ∈ {0,5 ; 2 ; 5 ; 10} × réduction du risque en régime de stress {aucune, légère, modérée, forte} (cible de volatilité × 1,00 / 0,90 / 0,75 / 0,60 en régime Stress). Critère : **ratio de Sharpe moyen des trois profils** sur 2020-2021. Le modèle de prévision éventuel est celui entraîné sur la seule période d'entraînement.
"""),
code(r"""
val_reb = [d for d in rebalance_dates(prices.index, pd.Timestamp(TRAIN_END) - pd.Timedelta(days=7),
                                      pd.Timestamp(VAL_END) - emb) if d in panel_dates]
fc_val_bundle = {**forecast_bundle, "pipeline": build_forecast_model(best_fc_name).fit(tr[FEATS], tr["target_reg"])}
gammas = [0.5, 2.0, 5.0, 10.0] if not QUICK_RUN else [2.0, 10.0]
mult_grid = REGIME_MULTIPLIER_GRID if not QUICK_RUN else {k: REGIME_MULTIPLIER_GRID[k] for k in ("aucune", "modérée")}
cal_rows = {}
for mult_name, mult in mult_grid.items():
    for gamma in gammas:
        for prof in PROFILE_CLASSES:
            Wv = run_multi_agent(val_reb, prof, {**regime_bundle, "vol_multiplier": mult},
                                 {**fc_val_bundle, "turnover_penalty": gamma})
            rv, to_v = simulate(Wv, returns.loc[:VAL_END])
            m = perf_metrics(rv, rf_daily, to_v)
            cal_rows[(mult_name, gamma, prof)] = {"Sharpe": m["Sharpe"], "Max drawdown": m["Max drawdown"],
                                                  "Rotation annuelle": m["Rotation annuelle"]}
cal_tab = pd.DataFrame(cal_rows).T
cal_tab.index.names = ["réduction du risque", "γ", "profil"]
cal_summary = cal_tab.groupby(level=["réduction du risque", "γ"]).mean().sort_values("Sharpe", ascending=False)
best_mult_name, TURNOVER_GAMMA = cal_summary["Sharpe"].idxmax()
TURNOVER_GAMMA = float(TURNOVER_GAMMA)
regime_bundle["vol_multiplier"] = mult_grid[best_mult_name]
regime_bundle["vol_multiplier_name"] = best_mult_name
forecast_bundle["turnover_penalty"] = TURNOVER_GAMMA
cal_tab.to_csv(RESULTS_DIR / "choix_reglages_validation.csv")
RESULTS["reglages_validation"] = cal_summary
display(cal_summary.style.format({"Sharpe": "{:.3f}", "Max drawdown": "{:.2%}", "Rotation annuelle": "{:.2f}"}))
print(f"Réglages retenus sur la validation : réduction du risque « {best_mult_name} » "
      f"{mult_grid[best_mult_name]} ; γ = {TURNOVER_GAMMA}")
"""),
md(r"""
### 7.3 Simulation sur le test
Les réglages sont maintenant **figés**. On calcule, à chaque fin de mois du test, les allocations de toutes les stratégies, puis on simule leur valeur jour par jour (les poids dérivent avec les prix entre deux rebalancements).
"""),
code(r"""
test_start = pd.Timestamp(VAL_END) + pd.Timedelta(days=1)
reb = [d for d in rebalance_dates(prices.index, test_start - pd.Timedelta(days=7), last_date) if d in panel_dates]
print(f"{len(reb)} rebalancements : {reb[0].date()} → {reb[-1].date()}")

yearly_fc = {}


def fc_bundle_at(d):
    if not FC_ACTIVE:
        return forecast_bundle
    if d.year not in yearly_fc:
        yearly_fc[d.year] = {**forecast_bundle, "pipeline": forecast_model_trained_until(pd.Timestamp(f"{d.year}-01-01"))}
    return yearly_fc[d.year]


W = {}
for prof in PROFILE_CLASSES:
    variants = [("Multi-agent", False)] + ([("Multi-agent sans ML", True)] if FC_ACTIVE else [])
    for variant, sans_ml in variants:
        orch_t = PortfolioOrchestrator(None, regime_bundle, None)
        W_v, prev_ = {}, None
        for d in reb:
            fb = fc_bundle_at(d)
            orch_t.forecast_bundle = {**fb, "ic": 0.0} if sans_ml else fb
            orch_t.turnover_penalty = TURNOVER_GAMMA
            r = orch_t.recommend(d, prices, panel, mkt_feat, profile=prof, prev_w=prev_, explain=False)
            W_v[d], prev_ = r["weights"], r["weights"]
        W[f"{variant} – {prof}"] = W_v

for d in reb:
    window = returns.loc[:d].tail(LOOKBACK_COV)
    avail = window.columns[window.notna().mean() >= 0.9].tolist()
    cov_d = ledoit_wolf_cov(window[avail])
    rp_d, hrp_d = risk_parity(cov_d), hrp(cov_d)
    W.setdefault("Équipondérée 1/N", {})[d] = equal_weight(avail)
    W.setdefault("60/40", {})[d] = sixty_forty(avail)
    W.setdefault("Variance minimale", {})[d] = min_variance(window[avail])
    W.setdefault("Markowitz historique", {})[d] = markowitz_historical(window[avail])
    for prof in PROFILE_CLASSES:
        tv = PROFILE_CONSTRAINTS[prof]["target_vol"]
        W.setdefault(f"Référence stratégique – {prof}", {})[d] = PortfolioOrchestrator.benchmark_weights(prof, avail)
        W.setdefault(f"1/N vol. cible – {prof}", {})[d] = vol_targeted_equal_weight(window[avail], tv)
        W.setdefault(f"Parité de risque vol. cible – {prof}", {})[d] = to_target_vol(rp_d, cov_d, tv)
        W.setdefault(f"HRP vol. cible – {prof}", {})[d] = to_target_vol(hrp_d, cov_d, tv)
print(f"{len(W)} stratégies simulées")
chrono("7a. Backtest : réglages et simulation")
"""),
]
