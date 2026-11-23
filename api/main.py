"""
API de l'assistant multi-agent (FastAPI).

Lancement depuis la racine du projet :
    .venv\\Scripts\\python.exe -m uvicorn api.main:app --port 8000
Documentation interactive : http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import services as sv

@asynccontextmanager
async def lifespan(_app):
    sv.state()          # charge les modèles et les données une seule fois, au démarrage
    yield


app = FastAPI(title="Assistant multi-agent de portefeuille", version="2.0", lifespan=lifespan,
              description="Agents Profil, Régime, Alerte, Prévision, Risque, Allocation et Explication. " + sv.DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/figures", StaticFiles(directory=str(sv.FIG_DIR)), name="figures")


class ClientForm(BaseModel):
    age: int = Field(40, ge=18, le=100)
    enfants: int = Field(0, ge=0, le=15)
    revenu: float = Field(70_000, ge=0)
    actifs: float = Field(300_000, ge=0)
    dettes: float = Field(100_000, ge=0)
    devise: str = "USD"
    taux_change: Optional[float] = Field(None, gt=0)
    connaissance: int = Field(5, ge=0, le=10)
    quiz: int = Field(2, ge=0, le=3)
    sexe: str = "Homme"
    etudes: str = "Diplôme universitaire"
    en_couple: bool = True
    situation: str = "Salarié"
    proprietaire: bool = True
    a_epargne: bool = True
    reponses: dict[str, int] = {}


class RecoRequest(ClientForm):
    profil_force: Optional[str] = None
    montant: float = Field(10_000, gt=0)


def _safe(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Artefact manquant : exécutez le notebook ({exc.filename}).")
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__} : {exc}")


@app.get("/api/sante")
def sante():
    s = sv.state()
    return {"statut": "ok", "date_donnees": str(s.last_date.date()), "modeles": s.meta.get("modele_profil")}


@app.get("/api/questionnaire")
def get_questionnaire():
    return sv.questionnaire()


@app.post("/api/profil")
def post_profil(form: ClientForm):
    return _safe(sv.profile, form.model_dump())


@app.post("/api/recommandation")
def post_recommandation(req: RecoRequest):
    d = req.model_dump()
    return _safe(sv.recommendation, d, d.pop("profil_force"), d.pop("montant"))


@app.get("/api/marche")
def get_marche(annees: int = 3):
    return _safe(sv.market, annees)


@app.get("/api/prix/{ticker}")
def get_prix(ticker: str, annees: int = 3):
    if ticker not in sv.UNIVERSE:
        raise HTTPException(404, "Actif inconnu")
    return _safe(sv.prices_series, ticker, annees)


@app.get("/api/backtest")
def get_backtest():
    return _safe(sv.backtest)


@app.get("/api/modeles")
def get_modeles():
    return _safe(sv.models_lab)


@app.get("/api/donnees")
def get_donnees():
    return _safe(sv.data_status)


@app.post("/api/donnees/maj")
def post_maj():
    try:
        return sv.refresh()
    except Exception as exc:     # rien n'est modifié en cas d'échec (cf. src/data_update.py)
        raise HTTPException(502, f"Mise à jour annulée, les données actuelles sont conservées. Motif : {exc}")


@app.post("/api/donnees/restaurer")
def post_restaurer():
    try:
        return sv.restore()
    except Exception as exc:
        raise HTTPException(400, f"Restauration impossible : {exc}")
