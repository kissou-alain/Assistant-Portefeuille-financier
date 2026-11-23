from cellules import code, md

CELLS = [
md(r"""
## 6. Système multi-agent : orchestration et allocation

### 6.1 Architecture (IA agentique)
Chaque agent est spécialisé, lit ce dont il a besoin sur un **tableau noir** (*blackboard*) partagé et y publie ses conclusions. Un **orchestrateur** fixe l'ordre d'intervention et gère une **boucle de négociation** entre l'agent Allocation et l'agent Risque.

```
Client  ──► Agent Profil ──────┐  profil retenu, alerte d'incohérence
Marché  ──► Agent Régime ──────┤  régime + probabilités → cible de risque ajustée
Marché  ──► Agent Alerte ──────┤  journée anormale ? (information du conseiller)
Panel   ──► Agent Prévision ───┤  rendements excédentaires prévus → α (0 si garde-fou)
Prix    ──► Agent Risque ──────┤  covariance de Ledoit-Wolf
                               ▼
              Agent Allocation ⇄ Agent Risque   (si une limite est dépassée : cible de volatilité × 0,9, jusqu'à 3 tours)
                               ▼
                       Agent Explication ──► recommandation justifiée + journal des échanges
```

### 6.2 L'agent Allocation : optimisation moyenne-variance avec a priori de Black-Litterman
L'agent résout, sous les contraintes du profil (poids maximal par ligne, plafond actions, part défensive minimale obligations + monétaire, plafond sur les obligations longues, budget de volatilité) :

$$\max_w\; w^T\mu \;-\; \frac{\lambda}{2}\,w^T\Sigma w \;-\; \frac{\gamma}{2}\,\lVert w - w_{préc}\rVert^2, \qquad \mu = \pi + \alpha, \qquad \pi = \lambda\,\Sigma\,w_{ref}$$

- $\Sigma$ : covariance estimée par **rétrécissement de Ledoit-Wolf** sur un an (plus stable que la covariance empirique avec 25 actifs) ;
- $\pi$ : rendements « d'équilibre » implicites de l'**allocation stratégique de référence** du profil $w_{ref}$. Sans signal, l'optimiseur retrouve exactement $w_{ref}$ (vérifié en 6.4) ;
- $\alpha$ : vues de l'agent Prévision, **nulles** ici puisque le garde-fou de 4.10 n'a pas activé le modèle ;
- $\lambda$ : aversion au risque du profil, augmentée en régime de stress ; $\gamma$ : pénalité de rotation (frein aux coûts de transaction).
"""),
code(r"""
from src.agents import PortfolioOrchestrator, AllocationAgent, RiskAgent
from src.allocation import ledoit_wolf_cov, risk_parity, hrp, to_target_vol

orchestrator = PortfolioOrchestrator(profile_bundle, regime_bundle, forecast_bundle, alert_bundle=alert_bundle)

# Clients fictifs : situation financière (variables du SCF, en dollars US) + réponses au questionnaire
# (indice de l'option choisie, de 0 = la plus prudente à 3 = la plus tolérante ; cf. src/questionnaire.py)
Q = lambda h, o, r, p, e, part, rev: {"Q_HORIZON": h, "Q_OBJECTIF": o, "Q_REACTION": r, "Q_PERTE": p,
                                     "Q_EXPERIENCE": e, "Q_PART": part, "Q_REVENUS": rev}
clients = {
    "A – retraité 68 ans": {"AGE": 68, "KIDS": 0, "INCOME": 45_000, "NETWORTH": 350_000, "ASSET": 380_000,
                            "DEBT": 30_000, "LEVRATIO": 0.08, "DEBT2INC": 0.6, "KNOWL": 5, "FINLIT": 2, "HHSEX": 1,
                            "EDCL": 2, "MARRIED": 1, "OCCAT1": 3, "LIFECL": 5, "HOUSECL": 1, "SAVED": 1,
                            **Q(1, 0, 1, 1, 1, 1, 2)},
    "B – cadre 41 ans": {"AGE": 41, "KIDS": 2, "INCOME": 110_000, "NETWORTH": 250_000, "ASSET": 520_000,
                         "DEBT": 270_000, "LEVRATIO": 0.52, "DEBT2INC": 2.4, "KNOWL": 7, "FINLIT": 3, "HHSEX": 1,
                         "EDCL": 4, "MARRIED": 1, "OCCAT1": 1, "LIFECL": 3, "HOUSECL": 1, "SAVED": 1,
                         **Q(2, 2, 2, 2, 2, 1, 2)},
    "C – entrepreneur 32 ans": {"AGE": 32, "KIDS": 0, "INCOME": 180_000, "NETWORTH": 600_000, "ASSET": 700_000,
                                "DEBT": 100_000, "LEVRATIO": 0.14, "DEBT2INC": 0.55, "KNOWL": 9, "FINLIT": 3, "HHSEX": 1,
                                "EDCL": 4, "MARRIED": 2, "OCCAT1": 2, "LIFECL": 1, "HOUSECL": 2, "SAVED": 1,
                                **Q(3, 3, 3, 3, 2, 2, 2)},
    # Cas d'incohérence : se déclare très tolérant, mais situation financière fragile
    "D – 26 ans, se déclare dynamique": {"AGE": 26, "KIDS": 1, "INCOME": 22_000, "NETWORTH": -8_000, "ASSET": 6_000,
                                         "DEBT": 14_000, "LEVRATIO": 2.3, "DEBT2INC": 0.64, "KNOWL": 2, "FINLIT": 1,
                                         "HHSEX": 2, "EDCL": 2, "MARRIED": 2, "OCCAT1": 1, "LIFECL": 1, "HOUSECL": 2,
                                         "SAVED": 0, **Q(3, 3, 3, 3, 1, 3, 2)},
}
last_date = panel.index.get_level_values("date").max()
demo = {name: orchestrator.recommend(last_date, prices, panel, mkt_feat, client=cl) for name, cl in clients.items()}
profil_tab = pd.DataFrame({n: {"Score questionnaire": f"{r['profile_detail']['questionnaire']['score']}/21",
                               "Tolérance déclarée": r["profile_detail"]["declare"],
                               "P(Prudent) – modèle": f"{r['profile_proba'].get('Prudent', np.nan):.0%}",
                               "Capacité (modèle)": r["profile_detail"]["capacite"],
                               "Profil retenu": r["profile"],
                               "Alerte conseiller": "oui" if r["profile_detail"]["alerte"] else "non",
                               "Vol. ex ante": f"{r['metrics']['vol_ex_ante']:.1%}"} for n, r in demo.items()}).T
profil_tab
"""),
md(r"""
### 6.3 Recommandation expliquée et journal des échanges entre agents
"""),
code(r"""
res_d = demo["D – 26 ans, se déclare dynamique"]
print(res_d["explanation"])
print("\nJournal des échanges (client D) :")
display(res_d["log"])

alloc_tab = pd.DataFrame({n: r["weights"] for n, r in demo.items()}).fillna(0)
alloc_tab = alloc_tab[(alloc_tab > 0.005).any(axis=1)]
alloc_tab.insert(0, "Classe", alloc_tab.index.map(ASSET_CLASS))
display(alloc_tab.sort_values("Classe").style.format({c: "{:.1%}" for c in demo}))
fig, axes = plt.subplots(1, len(demo), figsize=(5.5 * len(demo), 5))
for ax, (n, r) in zip(axes, demo.items()):
    ex = pd.Series(r["metrics"]["expositions"])
    ex = ex[ex > 0.005]
    ax.pie(ex, labels=ex.index, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"Client {n}\nProfil {r['profile']} | vol. {r['metrics']['vol_ex_ante']:.1%}")
plt.tight_layout()
savefig("17_demo_allocations")
plt.show()
"""),
md(r"""
**Lecture** : le client D se déclare dynamique (questionnaire), mais sa situation financière (revenu faible, patrimoine négatif, endettement) le rapproche des ménages prudents : le profil est ramené au plus prudent des deux et une **alerte** est adressée au conseiller. Le journal montre la séquence complète des échanges entre agents, y compris la décision du garde-fou de l'agent Prévision : ses prévisions sont affichées à titre indicatif, mais n'entrent pas dans l'allocation.

**À noter** : à la date de la démonstration, le marché est en régime **Stress** (volatilité élevée). La cible de risque de chaque profil est donc réduite, ce qui explique des volatilités ex ante inférieures aux cibles nominales (6 %, 10 %, 16 %). L'intensité de cette réduction sera choisie sur la période de validation en section 7.

### 6.4 Vérification de la propriété de Black-Litterman
Sans vue ($\alpha = 0$), avec l'aversion de base, une cible de volatilité non contraignante et sans portefeuille précédent, l'optimiseur doit retrouver **exactement** l'allocation de référence $w_{ref}$ : on vérifie l'écart maximal, en points de pourcentage, à plusieurs dates. En régime de stress (aversion plus forte, cible de volatilité plus basse), le portefeuille s'écarte au contraire **volontairement** de la référence, vers moins de risque.
"""),
code(r"""
bl_rows = []
for d in [pd.Timestamp("2019-06-28"), pd.Timestamp("2022-06-30"), last_date]:
    d = prices.index[prices.index <= d][-1]
    R = returns.loc[:d].tail(LOOKBACK_COV)
    R = R.loc[:, R.notna().mean() >= 0.9]
    cov_d = RiskAgent().estimate(R)
    for prof in PROFILE_CLASSES:
        c = PROFILE_CONSTRAINTS[prof]
        w_ref = PortfolioOrchestrator.benchmark_weights(prof, list(cov_d.index))
        pi = pd.Series(c["risk_aversion"] * cov_d.values @ w_ref.values, index=cov_d.index)
        w_a = AllocationAgent().optimize(pi, cov_d, c, 10.0, c["risk_aversion"], None)
        m_s = REGIME_MULTIPLIER_GRID["modérée"]["Stress"]
        w_b = AllocationAgent().optimize(pi, cov_d, c, c["target_vol"] * m_s, c["risk_aversion"] / m_s, None)
        bl_rows.append({"date": d.date(), "profil": prof, "vol. de la référence": float(np.sqrt(w_ref @ cov_d @ w_ref)),
                        "cible de vol.": c["target_vol"], "écart max (propriété)": (w_a - w_ref).abs().max(),
                        "écart max (régime Stress)": (w_b - w_ref).abs().max()})
bl_check = pd.DataFrame(bl_rows).set_index(["date", "profil"])
bl_check.to_csv(RESULTS_DIR / "verification_black_litterman.csv")
assert bl_check["écart max (propriété)"].max() < 0.005, "La propriété de Black-Litterman n'est pas vérifiée"
bl_check.style.format("{:.2%}")
"""),
md(r"""
### 6.5 Allocations de référence fondées uniquement sur le risque
Pour juger l'agent Allocation, on calcule à la même date trois allocations qui n'utilisent **aucune prévision de rendement** :
- **Parité de risque** (*Equal Risk Contribution*) : chaque actif contribue autant au risque total, $RC_i = w_i(\Sigma w)_i/\sigma_p$ égales.
- **HRP** (*Hierarchical Risk Parity*) : les actifs sont regroupés par **classification hiérarchique** (comme la CAH, sur la distance $d_{ij}=\sqrt{\tfrac12(1-\rho_{ij})}$), puis le risque est réparti récursivement entre les groupes. Robuste, car il n'inverse pas la matrice de covariance.
- **1/N** : poids égaux.

Chacune est ramenée à la volatilité cible du profil en la mélangeant avec la poche monétaire (SHY), pour comparer **à risque égal**.
"""),
code(r"""
R_last = returns.loc[:last_date].tail(LOOKBACK_COV)
cov_last = ledoit_wolf_cov(R_last)
refs = {"Parité de risque": risk_parity(cov_last), "HRP": hrp(cov_last),
        "1/N": pd.Series(1 / len(cov_last), index=cov_last.index)}
rows_ref = []
for prof in PROFILE_CLASSES:
    tv = PROFILE_CONSTRAINTS[prof]["target_vol"]
    r_ma = orchestrator.recommend(last_date, prices, panel, mkt_feat, profile=prof, explain=False)
    cands_w = {"Multi-agent": r_ma["weights"], **{k: to_target_vol(w, cov_last, tv) for k, w in refs.items()}}
    for nom, w in cands_w.items():
        w = w.reindex(cov_last.index).fillna(0)
        cls = w.groupby(w.index.map(ASSET_CLASS)).sum()
        rows_ref.append({"profil": prof, "stratégie": nom, "vol. ex ante": float(np.sqrt(w @ cov_last.values @ w)),
                         "Actions + immo.": cls.get("Actions", 0) + cls.get("Immobilier", 0),
                         "Obligations": cls.get("Obligations", 0), "Monétaire": cls.get("Monétaire", 0),
                         "Matières premières": cls.get("Matières premières", 0), "Crypto": cls.get("Crypto", 0),
                         "nb lignes > 1 %": int((w > 0.01).sum())})
ref_tab = pd.DataFrame(rows_ref).set_index(["profil", "stratégie"])
display(ref_tab.style.format({c: "{:.1%}" for c in ref_tab.columns if c != "nb lignes > 1 %"}))
chrono("6. Système multi-agent")
"""),
md(r"""
**Lecture** : à risque égal, les allocations fondées uniquement sur le risque (parité de risque, HRP) sont dominées par les obligations et le monétaire, qui sont les actifs les moins volatils. L'agent Allocation, qui part de l'allocation stratégique du profil, conserve davantage d'actions pour les profils Équilibré et Dynamique, conformément à leur objectif de croissance. Le backtest de la section 7 dira laquelle de ces approches a le mieux fonctionné hors échantillon.
"""),
]
