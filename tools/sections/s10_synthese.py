from cellules import code, md

CELLS = [
md(r"""
## 10. Synthèse des résultats et conclusion

La cellule suivante regroupe tous les tableaux de résultats dans `results/resultats_synthese.md`, utilisé pour le rapport et la présentation.
"""),
code(r"""
def to_md(df: pd.DataFrame) -> str:
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df.index = [" / ".join(map(str, i)) for i in df.index]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


titres = {
    "profil_validation": "Agent Profil – comparaison des modèles (validation)",
    "profil_desequilibre": "Agent Profil – corrections du déséquilibre (validation)",
    "profil_test": "Agent Profil – modèle retenu (test)",
    "profil_externe": "Agent Profil – validation externe (SCF 2019)",
    "prevision_validation": "Agent Prévision – comparaison des modèles (validation 2020-2021)",
    "prevision_trois_evaluations": "Agent Prévision – trois méthodes d'évaluation",
    "prevision_diebold_mariano": "Agent Prévision – prévisions hors pli 2013-2021 et tests de Diebold-Mariano",
    "prevision_test": "Agent Prévision – test (2022 → fin)",
    "regime_clustering": "Agent Régime – comparaison des 4 familles de clustering",
    "regime": "Agent Régime – validation économique",
    "prevision_apport_gamma": "Apport des probabilités du GMM comme variables (Diebold-Mariano)",
    "reglages_validation": "Réglages de l'allocation choisis sur la validation",
    "backtest": "Backtest – performances sur le test",
    "risque_comparable": "Backtest – comparaison à risque comparable",
    "significativite": "Backtest – significativité (bootstrap par blocs)",
    "rl_test": "Renforcement (PPO) – comparaison sur le test",
    "rl_significativite": "Renforcement (PPO) – significativité",
}
md_txt = [f"# Synthèse des résultats\n\nDonnées jusqu'au {last_date.date()} – mode rapide : {QUICK_RUN}\n\n"
          f"- Agent Profil : **{best_profile_name}** (seuil {THRESH:.2f})\n"
          f"- Agent Prévision : **{best_fc_name}** – {'activé' if FC_ACTIVE else 'non activé (garde-fou, α = 0)'} "
          f"(IC hors pli = {IC_VAL:.4f})\n"
          f"- Réglages : réduction du risque « {best_mult_name} », γ = {TURNOVER_GAMMA}\n"]
for key, titre in titres.items():
    if key in RESULTS:
        md_txt.append(f"\n## {titre}\n\n{to_md(RESULTS[key])}\n")
md_txt.append("\n## Temps de calcul par étape (s)\n\n" + to_md(pd.Series(DUREES, name="secondes").to_frame()) + "\n")
(RESULTS_DIR / "resultats_synthese.md").write_text("\n".join(md_txt), encoding="utf-8")
chrono("10. Synthèse")
print(f"Synthèse écrite dans {RESULTS_DIR / 'resultats_synthese.md'}")
print(f"Durée totale d'exécution du notebook : {(time.time() - _T0) / 60:.1f} min")
"""),
md(r"""
### Conclusion

**Ce que le système apporte.** Une chaîne complète, automatisée et déployée — données ouvertes → profil du client → lecture du régime de marché → alerte → prévisions → allocation sous contraintes → contrôle des risques → explication — évaluée sans fuite de données, avec des baselines, des tests statistiques et une ablation. Chaque famille de méthodes du cours y joue un rôle précis : classification supervisée (Profil), régression sur données temporelles (Prévision), réduction de dimension (ACP), clustering (Régime), détection d'anomalies (Alerte), renforcement (PPO), interprétabilité (SHAP).

**Ce que montrent les résultats.**
1. **Agent Profil** : la régression logistique, le modèle le plus simple, fait aussi bien que les ensembles d'arbres ; ses performances se maintiennent sur une autre vague de l'enquête (2019). La capacité à supporter des pertes s'explique bien par la situation financière.
2. **Agent Prévision** : malgré 13 modèles, une ACP et des variables de régime, **aucun modèle ne bat de façon significative la prévision nulle** au test de Diebold-Mariano, ni sur 2013-2021 ni sur le test. Le signal le plus fort de 2011-2021, la prime des actifs risqués, s'est retourné en 2022 : c'est une **dérive des données**. Le garde-fou décidé à l'avance a correctement empêché ces prévisions d'influencer l'allocation. La comparaison des trois méthodes d'évaluation montre aussi qu'un **hold-out unique aurait été trompeur**.
3. **Agent Régime** : les régimes repèrent bien le **risque** à venir (volatilité future plus élevée en régime Stress), mais pas la **direction** du marché ; réduire l'exposition en régime de stress n'a pas été retenu sur la validation. L'**agent Alerte** détecte toutes les crises connues.
4. **Allocation** : le risque réalisé respecte le budget de chaque profil. Pour les profils Équilibré et Dynamique, le multi-agent fait jeu égal avec les références simples de même risque ; pour le profil Prudent, il fait significativement moins bien que le 1/N et la parité de risque à volatilité cible.
5. **Renforcement** : l'agent PPO, prometteur sur la validation, ne se généralise pas au test (Sharpe négatif, significativement moins bon que les références) : surapprentissage dû au faible nombre de mois d'entraînement.

La valeur du système tient donc à l'**adéquation au profil**, au **contrôle du risque** et à l'**explicabilité** des recommandations, non à une surperformance.

Ces conclusions sont **cohérentes avec la littérature** sur l'efficience des marchés (DeMiguel et al., 2009 : la stratégie 1/N est difficile à battre hors échantillon).

**Limites**
- Période de test courte (≈ 5 ans) et déjà consultée lors des versions précédentes du projet ; tous les choix de cette version ont été figés sur la validation.
- Modèle de profil estimé sur des ménages américains ; à recalibrer avec un questionnaire local (MiFID II / AMMC) pour un déploiement au Maroc.
- Univers d'ETF cotés en dollars ; change et fiscalité non pris en compte ; coûts de transaction simplifiés.
- Détecteurs d'anomalies à recalibrer périodiquement (dérive après 2019).

**Perspectives** : réseaux récurrents (LSTM) sur des historiques plus longs ; apprentissage semi-supervisé pour exploiter des clients non étiquetés ; explication en langage naturel par un grand modèle de langage ; données alternatives (sentiment de presse).

> *Projet réalisé à des fins pédagogiques : les recommandations ne constituent pas un conseil en investissement.*
"""),
]
