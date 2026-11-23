"""
Exporte les résultats calculés par le code Python pour la version EN LIGNE du site (Vercel).

La version en ligne n'exécute pas Python : elle affiche les sorties produites ici par les mêmes
fonctions que l'API (api/services.py), à partir des modèles entraînés par le notebook.
Rien n'est ré-entraîné ni recalculé autrement.

Fichiers écrits dans web/public/donnees/ :
- marche.json, backtest.json, modeles.json, questionnaire.json, etat.json, prix.json ;
- portefeuilles.json : la recommandation pour chacun des 3 profils (le portefeuille ne dépend que
  du profil final, puisque l'agent Prévision est désactivé par le garde-fou) ;
- modele_profil.json : les PARAMÈTRES APPRIS du modèle de profil (imputation, écrêtage,
  standardisation, encodage one-hot, coefficients de la régression logistique, moyenne de fond
  SHAP). Le navigateur applique exactement la même formule (cf. web/src/lib/offline/profile.ts).

Usage :
    .venv\\Scripts\\python.exe tools\\exporter_site.py          # export seul
    .venv\\Scripts\\python.exe tools\\exporter_site.py --maj    # met d'abord à jour les données de marché
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "web" / "public" / "donnees"


def write(name: str, obj) -> None:
    from api.services import clean
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(clean(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {name:<22} {(OUT / name).stat().st_size / 1024:8.0f} Ko")


def export_profile_model(s) -> dict:
    """Paramètres appris du pipeline de profil : tout ce qu'il faut pour refaire le calcul ailleurs."""
    from src.explain import original_name
    pipe = s.profile["pipeline"]
    pre, logit = pipe.named_steps["pre"], pipe.named_steps["m"]
    num, cat = s.profile["num"], s.profile["cat"]
    tr = dict((name, (t, cols)) for name, t, cols in pre.transformers_)
    out_names = list(pre.get_feature_names_out())

    mt, money = tr["montants"]
    nt, other = tr["num"]
    ct, cats = tr["cat"]
    ohe = ct.named_steps["ohe"]
    # Encodage one-hot : pour chaque variable catégorielle, quelle colonne de sortie vaut 1 pour chaque modalité
    # (les modalités rares sont regroupées ; une modalité inconnue donne des zéros). On le mesure directement
    # avec l'encodeur appris, pour être certain de reproduire son comportement.
    base = len(money) + len(other)
    cat_maps = {}
    offset = 0
    for i, c in enumerate(cats):
        n_out = len([n for n in ohe.get_feature_names_out(cats) if n.startswith(f"{c}_")])
        mapping = {}
        for v in ohe.categories_[i]:
            row = [[v if cc == c else ohe.categories_[j][0] for j, cc in enumerate(cats)]]
            vec = ohe.transform(np.array(row, dtype=object))[0][offset:offset + n_out]
            mapping[str(v)] = int(np.argmax(vec)) if vec.max() > 0 else -1
        cat_maps[c] = {"offset": base + offset, "size": n_out, "mapping": mapping,
                       "imputation": str(ct.named_steps["imp"].statistics_[i])}
        offset += n_out

    exp = s.explainer
    return {
        "ordre_colonnes": out_names,
        "groupes": [original_name(n, num + cat) for n in out_names],
        "montants": {"variables": list(money), "mediane": mt.named_steps["imp"].statistics_.tolist(),
                     "moyenne": mt.named_steps["sc"].mean_.tolist(), "ecart_type": mt.named_steps["sc"].scale_.tolist()},
        "numeriques": {"variables": list(other), "mediane": nt.named_steps["imp"].statistics_.tolist(),
                       "bas": nt.named_steps["win"].low_.tolist(), "haut": nt.named_steps["win"].high_.tolist(),
                       "moyenne": nt.named_steps["sc"].mean_.tolist(), "ecart_type": nt.named_steps["sc"].scale_.tolist()},
        "categorielles": cat_maps,
        "coefficients": logit.coef_[0].tolist(), "constante": float(logit.intercept_[0]),
        "shap_moyenne_fond": np.asarray(exp.mean).tolist(),
        "seuil": s.profile["threshold"], "classes": list(s.profile["classes"]),
        "modele": s.profile.get("model_name"),
    }


def main(update: bool) -> None:
    if update:
        from src import data_update
        try:
            last = data_update.refresh_market_data()
            print(f"Données de marché mises à jour jusqu'au {last:%d/%m/%Y}")
        except Exception as exc:          # Yahoo/FRED indisponibles : on exporte les données actuelles
            print(f"Mise à jour impossible ({exc}) : export des données actuelles")
    from api import services as sv
    from src.config import PROFILE_CLASSES, SCF_LABELS, UNIVERSE, CURRENCIES
    from src.questionnaire import CAPS, QUESTIONS, THRESHOLDS
    s = sv.state()
    print(f"Export du site (données au {s.last_date:%d/%m/%Y}) vers {OUT}")
    write("etat.json", {**sv.data_status(), "mode": "en ligne"})
    write("questionnaire.json", sv.questionnaire())
    write("marche.json", sv.market())
    write("prix.json", {t: sv.prices_series(t) for t in UNIVERSE})
    write("backtest.json", sv.backtest())
    write("modeles.json", sv.models_lab())
    # Figures du laboratoire (servies par l'API en local, par le site lui-même en ligne)
    import shutil
    fig_out = OUT.parent / "figures"
    fig_out.mkdir(parents=True, exist_ok=True)
    for fid in sv.FIGURES:
        src = sv.FIG_DIR / f"{fid}.png"
        if src.exists():
            shutil.copy2(src, fig_out / src.name)
    print(f"  {len(list(fig_out.glob('*.png')))} figures copiées dans {fig_out}")
    # Une recommandation par profil final (client neutre : seule la partie « profil » change selon le visiteur,
    # elle est recalculée dans le navigateur avec modele_profil.json)
    write("portefeuilles.json", {p: sv.recommendation({}, forced_profile=p, montant=100_000) for p in PROFILE_CLASSES})
    write("modele_profil.json", {
        **export_profile_model(s), "libelles": SCF_LABELS,
        "questionnaire": {"questions": list(QUESTIONS), "seuils": THRESHOLDS,
                          "plafonds": [{"question": k, "reponse": v, "profil": p} for (k, v), p in CAPS.items()],
                          "libelles": {k: opts for k, (_, opts) in QUESTIONS.items()}},
        "devises": CURRENCIES,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maj", action="store_true", help="mettre à jour les données de marché avant l'export")
    main(ap.parse_args().maj)
