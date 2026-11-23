from cellules import code, md

CELLS = [
md(r"""
### 7.4 Performances sur le test
Mesures : rendement et volatilité annualisés, **ratio de Sharpe** (rendement excédentaire au taux sans risque / volatilité), **Sortino** (ne pénalise que la volatilité à la baisse), **perte maximale** (*max drawdown*), **Calmar** (rendement / perte maximale), VaR journalière à 95 %, **rotation annuelle** (somme des poids échangés par an).
"""),
code(r"""
bt_returns, bt_rows = {}, {}
for name, wd in W.items():
    r, turnover = simulate(wd, returns)
    bt_returns[name] = r
    bt_rows[name] = perf_metrics(r, rf_daily, turnover)
bt_returns = pd.DataFrame(bt_returns)
bt_table = pd.DataFrame(bt_rows).T
bt_table.to_csv(RESULTS_DIR / "backtest_performances_test.csv")
bt_returns.to_csv(RESULTS_DIR / "backtest_rendements_quotidiens.csv")
RESULTS["backtest"] = bt_table
ratio_cols = ("Sharpe", "Sortino", "Calmar", "Rotation annuelle")
fmt = {c: "{:.2%}" for c in bt_table.columns if c not in ratio_cols} | {c: "{:.2f}" for c in ratio_cols}
display(bt_table[[c for c in bt_table.columns]].style.format(fmt).background_gradient(subset=["Sharpe"], cmap="RdYlGn"))
"""),
md(r"""
**Comparaison à risque égal, profil par profil** : c'est le tableau qui permet de juger le système.
"""),
code(r"""
cols_rc = ["Rendement annualisé", "Volatilité annualisée", "Sharpe", "Max drawdown", "Rotation annuelle"]
strat_par_profil = ["Multi-agent", "Multi-agent sans ML", "Référence stratégique", "1/N vol. cible",
                    "Parité de risque vol. cible", "HRP vol. cible"]
rc = []
for prof in PROFILE_CLASSES:
    for s in strat_par_profil:
        key = f"{s} – {prof}"
        if key in bt_table.index:
            rc.append(bt_table.loc[key, cols_rc].rename((prof, s)))
risk_comp = pd.DataFrame(rc)
risk_comp.index = pd.MultiIndex.from_tuples(risk_comp.index, names=["profil", "stratégie"])
risk_comp.to_csv(RESULTS_DIR / "backtest_risque_comparable.csv")
RESULTS["risque_comparable"] = risk_comp
risk_comp.style.format({k: v for k, v in fmt.items() if k in risk_comp.columns})
"""),
code(r"""
wealth = (1 + bt_returns.fillna(0)).cumprod()
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
for ax, prof in zip(axes, PROFILE_CLASSES):
    for s, style in (("Multi-agent", "-"), ("Référence stratégique", ":"), ("1/N vol. cible", "--"),
                     ("Parité de risque vol. cible", "-."), ("HRP vol. cible", (0, (1, 1)))):
        wealth[f"{s} – {prof}"].plot(ax=ax, ls=style, lw=2 if s == "Multi-agent" else 1.3, label=s)
    ax.set_title(f"Profil {prof} – à risque comparable")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("18b_backtest_risque_comparable")
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
bench = ["Équipondérée 1/N", "60/40", "Variance minimale", "Markowitz historique"]
for c in bench:
    wealth[c].plot(ax=axes[0], ls="--", lw=1.1)
for prof in PROFILE_CLASSES:
    wealth[f"Multi-agent – {prof}"].plot(ax=axes[0], lw=2)
axes[0].set_title("Valeur du portefeuille (base 1) – période de test")
axes[0].legend(ncol=2)
dd = wealth / wealth.cummax() - 1
for c in bench[:2] + [f"Multi-agent – {p}" for p in PROFILE_CLASSES]:
    dd[c].plot(ax=axes[1], lw=1.2)
axes[1].set_title("Pertes depuis le plus haut (drawdowns)")
plt.tight_layout()
savefig("18_backtest_valeur_drawdown")
plt.show()
"""),
code(r"""
wq = pd.DataFrame(W["Multi-agent – Équilibré"]).T.fillna(0)
wq_cls = wq.T.groupby(wq.columns.map(ASSET_CLASS)).sum().T
fig, axes = plt.subplots(1, 2, figsize=(17, 5))
wq_cls.plot.area(ax=axes[0], linewidth=0)
axes[0].set_ylim(0, 1)
axes[0].set_title("Allocation par classe – Multi-agent Équilibré")
reg_at = reg_series.reindex(wq.index, method="ffill")
axes[1].bar(range(len(reg_at)), 1, color=reg_at.map(colors), width=1)
axes[1].set_xticks(range(0, len(reg_at), 6))
axes[1].set_xticklabels([d.strftime("%Y-%m") for d in reg_at.index[::6]], rotation=45)
axes[1].set_yticks([])
axes[1].set_title("Régime à chaque rebalancement (vert = Calme, jaune = Neutre, rouge = Stress)")
plt.tight_layout()
savefig("19_backtest_allocations")
plt.show()
"""),
md(r"""
### 7.5 Significativité statistique : bootstrap par blocs
Sur 4 à 5 ans, les écarts de ratio de Sharpe sont très bruités. On estime un intervalle de confiance à 95 % de la **différence de Sharpe** entre le système et chaque référence de même risque, par **bootstrap par blocs** : on rééchantillonne des blocs de 21 jours consécutifs, ce qui préserve l'autocorrélation et le regroupement de volatilité. Si l'intervalle contient 0, la différence n'est **pas statistiquement établie**.
"""),
code(r"""
def block_bootstrap_sharpe_diff(r1, r2, n_boot=2000 if not QUICK_RUN else 300, block=21, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    pair = pd.concat([r1, r2], axis=1).dropna()
    df = pair.sub(rf_daily.reindex(pair.index).fillna(0), axis=0).values      # rendements excédentaires
    n = len(df)
    nb = int(np.ceil(n / block))
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block, nb)])[:n]
        s = df[idx]
        sh = s.mean(0) / s.std(0) * np.sqrt(252)
        diffs.append(sh[0] - sh[1])
    diffs = np.array(diffs)
    obs = df.mean(0) / df.std(0) * np.sqrt(252)
    return {"diff. de Sharpe observée": obs[0] - obs[1], "IC 95 % bas": np.quantile(diffs, .025),
            "IC 95 % haut": np.quantile(diffs, .975), "P(diff ≤ 0)": (diffs <= 0).mean()}


sig = {}
for p in PROFILE_CLASSES:
    for ref in [s for s in strat_par_profil[1:] if f"{s} – {p}" in bt_returns]:
        sig[f"{p} : Multi-agent vs {ref}"] = block_bootstrap_sharpe_diff(bt_returns[f"Multi-agent – {p}"],
                                                                        bt_returns[f"{ref} – {p}"])
sig = pd.DataFrame(sig).T
sig["significatif (5 %)"] = np.where((sig["IC 95 % bas"] > 0) | (sig["IC 95 % haut"] < 0), "oui", "non")
sig.to_csv(RESULTS_DIR / "backtest_significativite.csv")
RESULTS["significativite"] = sig
display(sig)
chrono("7b. Backtest : résultats et significativité")
"""),
]
