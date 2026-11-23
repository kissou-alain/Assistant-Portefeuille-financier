from cellules import code, md

CELLS = [
md(r"""
## 9. Sauvegarde des artefacts pour l'application

Étape 6 de la méthode du cours : une fois l'évaluation terminée et les choix figés, le modèle à **déployer** peut être ré-entraîné sur **toutes** les données disponibles (plus de données → estimation plus précise), avec les hyperparamètres et le seuil déjà choisis.
- **Agent Profil** : régression logistique ré-entraînée sur toute l'enquête SCF 2022 (même seuil de décision).
- **Agent Prévision** : modèle ré-entraîné avec toutes les cibles connues à la date des données (garde-fou inchangé).
- **Agents Régime et Alerte** : modèles ajustés sur 2011-2019, conservés tels quels (ce sont eux qui ont été validés).
"""),
code(r"""
profile_final = clone(profile_bundle["pipeline"]).fit(X_p, y_int)
joblib.dump({**profile_bundle, "pipeline": profile_final}, MODELS_DIR / "agent_profil.joblib")
joblib.dump(regime_bundle, MODELS_DIR / "agent_regime.joblib")
joblib.dump(alert_bundle, MODELS_DIR / "agent_alerte.joblib")
fc_final = {**forecast_bundle, "pipeline": forecast_model_trained_until(last_date + pd.Timedelta(days=366))}
joblib.dump(fc_final, MODELS_DIR / "agent_prevision.joblib")

recent_dates = panel.index.get_level_values("date").unique().sort_values()[-260:]
panel.loc[panel.index.get_level_values("date").isin(recent_dates)].to_csv(DATA_PROC / "panel_recent.csv")
mkt_feat.to_csv(DATA_PROC / "market_features.csv")
proba_all.to_csv(DATA_PROC / "regimes_probabilites.csv")
flags.to_csv(DATA_PROC / "anomalies.csv")

meta = {"date_donnees": str(last_date.date()), "mode_rapide": QUICK_RUN,
        "modele_profil": best_profile_name, "seuil_profil": THRESH,
        "modele_prevision": best_fc_name, "prevision_active": FC_ACTIVE, "IC_hors_pli": IC_VAL, "IC_utilise": IC_USED,
        "penalite_rotation": TURNOVER_GAMMA, "reduction_risque_regime": best_mult_name,
        "variables_prevision": FEATS, "top_variables": TOP_FEATURES, "alerte_min_votes": ALERT_MIN,
        "rl_graines_retenues": {p: int(best_seed[p][1]) for p in PROFILE_CLASSES}}
(MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print("Artefacts enregistrés :", sorted(p.name for p in MODELS_DIR.iterdir() if p.suffix in (".joblib", ".json", ".zip")))
chrono("9. Sauvegarde")
"""),
]
