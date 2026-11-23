"""
Test d'égalité : la version en ligne (calcul dans le navigateur, web/src/lib/offline) donne-t-elle
EXACTEMENT les mêmes réponses que l'API Python ?

1. génère des clients aléatoires variés (devises, montants, situations, questionnaire complet, partiel ou absent) ;
2. calcule la réponse Python (api/services.profile et recommendation) ;
3. exécute le code TypeScript du site avec Node (tools/test_profil_parite.mjs) sur les mêmes clients ;
4. compare : probabilité, profils, alerte, explication SHAP, texte de la recommandation, journal, projection.

Prérequis : avoir lancé tools/exporter_site.py (mêmes modèles). Usage :
    .venv\\Scripts\\python.exe tools\\test_profil_parite.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import services as sv                        # noqa: E402
from api.main import ClientForm, RecoRequest           # noqa: E402
from src.questionnaire import QUESTIONS                # noqa: E402

N_PROFILS, N_RECOS, TOL = 200, 25, 1e-9


def random_form(rng: np.random.Generator) -> dict:
    devise = rng.choice(["USD", "MAD", "EUR"])
    k = {"USD": 1, "MAD": 10, "EUR": 0.9}[devise]
    f = {
        "age": int(rng.integers(18, 90)), "enfants": int(rng.integers(0, 5)),
        "revenu": float(np.round(rng.lognormal(11, 1) * k, 2)) * (rng.random() > 0.05),
        "actifs": float(np.round(rng.lognormal(12, 1.8) * k, 2)) * (rng.random() > 0.1),
        "dettes": float(np.round(rng.lognormal(10.5, 1.5) * k, 2)) * (rng.random() > 0.3),
        "devise": devise, "taux_change": None if rng.random() < 0.5 else float(rng.uniform(0.5, 12)),
        "connaissance": int(rng.integers(0, 11)), "quiz": int(rng.integers(0, 4)),
        "sexe": rng.choice(["Homme", "Femme"]),
        "etudes": rng.choice(["Sans diplôme du secondaire", "Diplôme du secondaire", "Études supérieures courtes",
                              "Diplôme universitaire"]),
        "en_couple": bool(rng.random() < 0.5),
        "situation": rng.choice(["Salarié", "Indépendant / chef d'entreprise", "Retraité", "Sans activité"]),
        "proprietaire": bool(rng.random() < 0.5), "a_epargne": bool(rng.random() < 0.6),
    }
    u = rng.random()
    keys = list(QUESTIONS)
    if u < 0.2:
        f["reponses"] = {}
    elif u < 0.4:
        f["reponses"] = {q: int(rng.integers(0, 4)) for q in rng.choice(keys, size=int(rng.integers(1, len(keys))), replace=False)}
    else:
        f["reponses"] = {q: int(rng.integers(0, 4)) for q in keys}
    return {key: (v.item() if isinstance(v, np.generic) else v) for key, v in f.items()}


def main() -> int:
    rng = np.random.default_rng(2026)
    forms = [random_form(rng) for _ in range(N_PROFILS)]
    forms[0] = {}                                       # formulaire vide : valeurs par défaut
    for f in forms[1::2]:                               # un client sur deux aisé et averti → profils non prudents
        f.update(revenu=f["revenu"] * 8, actifs=f["actifs"] * 40, connaissance=int(rng.integers(6, 11)), quiz=3,
                 etudes="Diplôme universitaire", a_epargne=True, age=int(rng.integers(30, 60)))
        if f["reponses"]:
            f["reponses"] = {q: int(rng.integers(1, 4)) for q in f["reponses"]}
    cases = []
    for i, f in enumerate(forms):
        case = {"form": f, "profil": sv.profile(ClientForm(**f).model_dump())}
        if i < N_RECOS:
            body = {**f, "montant": float(rng.choice([5_000, 100_000, 1_234_567])),
                    "profil_force": None if i % 5 else rng.choice(["Prudent", "Équilibré", "Dynamique"]).item()}
            d = RecoRequest(**body).model_dump()
            case["body"] = body
            case["reco"] = sv.recommendation(d, d.pop("profil_force"), d.pop("montant"))
        cases.append(case)
    tmp = Path(tempfile.mkdtemp()) / "cas.json"
    tmp.write_text(json.dumps(sv.clean(cases), ensure_ascii=False), encoding="utf-8")
    r = subprocess.run(["node", "--no-warnings", str(ROOT / "tools" / "test_profil_parite.mjs"), str(tmp), str(TOL)],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    print(r.stdout, r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
