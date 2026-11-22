"""
Questionnaire de tolérance au risque déclarée (inspiré des questionnaires
d'adéquation MiFID II / recommandations AMMC).

La réglementation distingue :
- la TOLÉRANCE au risque déclarée par le client (ce questionnaire, barème à points) ;
- sa CAPACITÉ à supporter des pertes, estimée ici par le modèle ML de l'agent
  Profil à partir de sa situation financière (données SCF).

Le profil retenu est le plus prudent des deux ; une divergence est signalée au
conseiller. Le barème n'est pas appris (aucune base de données publique ne
contient les réponses à ce questionnaire) : il est fixé a priori et documenté.
"""
from __future__ import annotations

PROFILE_ORDER = ["Prudent", "Équilibré", "Dynamique"]

# clé -> (intitulé, [options dans l'ordre croissant de tolérance au risque : 0, 1, 2, 3 points])
QUESTIONS = {
    "Q_HORIZON": ("Dans combien de temps aurez-vous besoin de cet argent ?",
                  ["Moins de 2 ans", "2 à 5 ans", "5 à 10 ans", "Plus de 10 ans"]),
    "Q_OBJECTIF": ("Quel est votre objectif principal ?",
                   ["Préserver mon capital", "Obtenir des revenus réguliers",
                    "Faire croître mon capital de façon équilibrée", "Maximiser la croissance à long terme"]),
    "Q_REACTION": ("Votre portefeuille perd 20 % en quelques mois. Que faites-vous ?",
                   ["Je vends tout", "Je vends une partie", "J'attends sans rien changer",
                    "J'investis davantage"]),
    "Q_PERTE": ("Quelle baisse annuelle maximale pourriez-vous supporter sans vendre ?",
                ["Moins de 5 %", "5 à 10 %", "10 à 20 %", "Plus de 20 %"]),
    "Q_EXPERIENCE": ("Quelle est votre expérience des placements financiers ?",
                     ["Aucune", "Comptes d'épargne et fonds prudents",
                      "Actions et obligations", "Investisseur régulier, y compris produits complexes"]),
    "Q_PART": ("Quelle part de votre patrimoine représente ce placement ?",
               ["Plus de 50 %", "25 à 50 %", "10 à 25 %", "Moins de 10 %"]),
    "Q_REVENUS": ("Vos revenus des 5 prochaines années sont-ils :",
                  ["Incertains", "Plutôt instables", "Stables", "Très stables et en croissance"]),
}
MAX_SCORE = 3 * len(QUESTIONS)
# Seuils sur le score total (0 à 21)
THRESHOLDS = {"Équilibré": 8, "Dynamique": 15}
# Réponses éliminatoires : plafonnent le profil quel que soit le score
CAPS = {
    ("Q_HORIZON", 0): "Prudent",      # horizon < 2 ans
    ("Q_PERTE", 0): "Prudent",        # perte acceptable < 5 %
    ("Q_REACTION", 0): "Prudent",     # vendrait tout après une baisse
    ("Q_HORIZON", 1): "Équilibré",    # horizon 2-5 ans : pas de profil dynamique
}


def has_answers(client: dict) -> bool:
    return any(k in client for k in QUESTIONS)


def most_prudent(*profiles: str) -> str:
    return min((p for p in profiles if p), key=PROFILE_ORDER.index)


def score_questionnaire(answers: dict) -> dict:
    """answers : {clé de question: indice de l'option choisie (0 à 3)}.
    Les questions sans réponse sont comptées comme la réponse la plus prudente."""
    detail, total, caps = {}, 0, []
    for k in QUESTIONS:
        v = answers.get(k)
        pts = int(v) if v is not None and 0 <= int(v) <= 3 else 0
        detail[k] = pts
        total += pts
        cap = CAPS.get((k, pts))
        if cap:
            caps.append(f"{QUESTIONS[k][1][pts]} → profil plafonné à {cap}")
    if total >= THRESHOLDS["Dynamique"]:
        profile = "Dynamique"
    elif total >= THRESHOLDS["Équilibré"]:
        profile = "Équilibré"
    else:
        profile = "Prudent"
    cap_profiles = [CAPS[(k, p)] for k, p in detail.items() if (k, p) in CAPS]
    if cap_profiles:
        profile = most_prudent(profile, *cap_profiles)
    return {"score": total, "score_max": MAX_SCORE, "profile": profile,
            "detail": detail, "plafonds": caps}
