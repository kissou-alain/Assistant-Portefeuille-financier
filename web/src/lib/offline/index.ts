// Mode en ligne (site statique Vercel) : mêmes réponses que l'API FastAPI, lues dans les fichiers
// exportés par tools/exporter_site.py (web/public/donnees). Seul le profil (et la partie « profil » de la
// recommandation) dépend du visiteur : il est calculé dans le navigateur avec les paramètres appris.
import { profileFromForm } from "./profile.ts";   // extension explicite : le test de parité l'exécute avec Node

const cache = new Map<string, Promise<any>>();
function file(name: string): Promise<any> {
  if (!cache.has(name)) {
    cache.set(name, fetch(`/donnees/${name}`).then((r) => {
      if (!r.ok) throw new Error(`Fichier ${name} introuvable`);
      return r.json();
    }).catch((e) => { cache.delete(name); throw e; }));
  }
  return cache.get(name)!;
}

// Valeurs par défaut du formulaire (= ClientForm dans api/main.py)
const DEFAULTS = {
  age: 40, enfants: 0, revenu: 70_000, actifs: 300_000, dettes: 100_000, devise: "USD", taux_change: null,
  connaissance: 5, quiz: 2, sexe: "Homme", etudes: "Diplôme universitaire", en_couple: true, situation: "Salarié",
  proprietaire: true, a_epargne: true, reponses: {},
};
const withDefaults = (b: any) => {
  const f: any = { ...DEFAULTS };
  for (const [k, v] of Object.entries(b ?? {})) if (v !== undefined && v !== null) f[k] = v;
  return f;
};

// Quantiles de la loi normale utilisés par la projection (= scipy norm.ppf)
const Z: Record<string, number> = { q5: -1.6448536269514729, q25: -0.6744897501960817, q50: 0, q75: 0.6744897501960817, q95: 1.6448536269514729 };
function projection(mean: number, vol: number, montant: number) {
  return Array.from({ length: 11 }, (_, t) => {
    const row: any = { annee: t };
    for (const [k, z] of Object.entries(Z)) row[k] = montant * Math.exp((mean - vol ** 2 / 2) * t + z * vol * Math.sqrt(t));
    return row;
  });
}

async function recommendation(body: any) {
  const [ports, model] = await Promise.all([file("portefeuilles.json"), file("modele_profil.json")]);
  const forced = ["Prudent", "Équilibré", "Dynamique"].includes(body?.profil_force) ? body.profil_force : null;
  const montant = Number(body?.montant ?? 10_000);
  const p = forced ? null : profileFromForm(withDefaults(body), model);
  const profil = forced ?? p!.profil;
  const r = structuredClone(ports[profil]);
  r.projection = { ...r.projection, montant, points: projection(r.projection.rendement_espere, r.projection.volatilite, montant) };
  if (!p) return r;

  // Partie « profil » propre au visiteur (mêmes phrases que src/agents.py)
  r.profil_detail = { declare: p.declare, capacite: p.capacite, final: p.profil, alerte: p.alerte };
  const lignes = [`**Profil investisseur retenu** : ${p.profil} – tolérance déclarée : ${p._texte.decl} ; `
    + `capacité estimée par le modèle : ${p.capacite} (probabilité de profil prudent ${p._texte.probaTxt}).`];
  if (p.alerte) lignes.push(`⚠️ **${p.alerte}**`);
  r.explication = r.explication.split("\n").flatMap((l: string) => (l.startsWith("**Profil investisseur") ? lignes : [l])).join("\n");
  const t = r.journal.find((e: any) => e.sender === "Agent Profil")?.timestamp ?? "";
  const entree = (key: string, summary: string) => ({ sender: "Agent Profil", key, summary, timestamp: t });
  const profilJournal = [
    entree("profile_detail", `Tolérance déclarée : ${p._texte.declTxt} | Capacité (ML) : ${p._texte.capTxt}`),
    ...(p.alerte ? [entree("profile_alert", p.alerte)] : []),
    entree("profile", `Profil retenu : ${p.profil}`),
    entree("profile_proba", "profile_proba"),
  ];
  r.journal = [...profilJournal, ...r.journal.filter((e: any) => e.sender !== "Agent Profil")];
  return r;
}

export class OfflineError extends Error {}

export async function offlineApi(path: string, body?: unknown): Promise<any> {
  const route = path.split("?")[0];
  if (route === "/api/profil") {
    const { _texte, ...res } = profileFromForm(withDefaults(body), await file("modele_profil.json"));
    return res;
  }
  if (route === "/api/recommandation") return recommendation(body);
  if (route.startsWith("/api/prix/")) {
    const t = decodeURIComponent(route.slice("/api/prix/".length));
    const prix = await file("prix.json");
    if (!(t in prix)) throw new OfflineError("Actif inconnu");
    return prix[t];
  }
  if (route === "/api/donnees/maj" || route === "/api/donnees/restaurer")
    throw new OfflineError("Dans la version en ligne, les données sont mises à jour automatiquement chaque soir de semaine "
      + "(après la clôture de New York). Ce bouton fonctionne dans la version locale.");
  const files: Record<string, string> = {
    "/api/questionnaire": "questionnaire.json", "/api/marche": "marche.json", "/api/backtest": "backtest.json",
    "/api/modeles": "modeles.json", "/api/donnees": "etat.json",
  };
  if (route in files) return file(files[route]);
  if (route === "/api/sante") {
    const e = await file("etat.json");
    return { statut: "ok", mode: "en ligne", date_donnees: e.date_donnees ?? null };
  }
  throw new OfflineError(`Route inconnue : ${route}`);
}
