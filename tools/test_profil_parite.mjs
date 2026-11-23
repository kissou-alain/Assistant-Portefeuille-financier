// Partie Node du test de parité (lancée par tools/test_profil_parite.py) : exécute le code TypeScript du site
// (web/src/lib/offline) sur les clients générés par Python et compare aux réponses de l'API Python.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [, , casesPath, tolArg] = process.argv;
const TOL = Number(tolArg ?? 1e-9);
const ROOT = process.cwd();
const PUBLIC = join(ROOT, "web", "public");

// Le site lit /donnees/*.json : on sert les fichiers exportés localement
globalThis.fetch = async (url) => {
  const body = readFileSync(join(PUBLIC, String(url)), "utf-8");
  return { ok: true, json: async () => JSON.parse(body) };
};
const { offlineApi } = await import(pathToFileURL(join(ROOT, "web", "src", "lib", "offline", "index.ts")).href);

const cases = JSON.parse(readFileSync(casesPath, "utf-8"));
const errors = [];
let maxDiff = 0;
const close = (a, b, where, rel = false) => {
  const d = Math.abs(a - b) / (rel ? Math.max(1, Math.abs(b)) : 1);
  maxDiff = Math.max(maxDiff, d);
  if (!(d <= TOL)) errors.push(`${where} : navigateur ${a} ≠ Python ${b}`);
};
const same = (a, b, where) => {
  if (JSON.stringify(a) !== JSON.stringify(b)) errors.push(`${where} :\n    navigateur ${JSON.stringify(a)}\n    Python     ${JSON.stringify(b)}`);
};

let nReco = 0;
for (const [i, c] of cases.entries()) {
  const js = await offlineApi("/api/profil", c.form), py = c.profil;
  close(js.proba_prudent, py.proba_prudent, `client ${i} proba_prudent`);
  for (const k of ["profil", "declare", "capacite", "alerte", "seuil", "questionnaire", "modele"]) same(js[k], py[k], `client ${i} ${k}`);
  close(js.explication.valeur_de_base, py.explication.valeur_de_base, `client ${i} valeur de base SHAP`);
  same(js.explication.unite, py.explication.unite, `client ${i} unité`);
  same(js.explication.contributions.map((x) => [x["libellé"], x.effet]), py.explication.contributions.map((x) => [x["libellé"], x.effet]), `client ${i} variables SHAP`);
  js.explication.contributions.forEach((x, j) => close(x.contribution, py.explication.contributions[j]?.contribution, `client ${i} SHAP ${x["libellé"]}`));

  if (c.reco) {
    nReco++;
    const a = await offlineApi("/api/recommandation", c.body), b = c.reco;
    for (const k of ["date", "profil", "profil_detail", "regime", "alerte_marche", "classes", "explication", "devise_note", "avertissement"]) same(a[k], b[k], `reco ${i} ${k}`);
    same(a.lignes.map((l) => [l.ticker, l.poids]), b.lignes.map((l) => [l.ticker, l.poids]), `reco ${i} lignes`);
    same(a.journal.map((e) => [e.sender, e.key, e.summary]), b.journal.map((e) => [e.sender, e.key, e.summary]), `reco ${i} journal`);
    same(a.projection.montant, b.projection.montant, `reco ${i} montant`);
    a.projection.points.forEach((p, t) => ["q5", "q25", "q50", "q75", "q95"].forEach((q) => close(p[q], b.projection.points[t][q], `reco ${i} projection année ${t} ${q}`, true)));
  }
}

const nAlert = cases.filter((c) => c.profil.alerte).length;
const dist = cases.reduce((m, c) => ((m[c.profil.profil] = (m[c.profil.profil] ?? 0) + 1), m), {});
console.log(`${cases.length} clients (profils ${JSON.stringify(dist)}, ${nAlert} alertes d'incohérence), ${nReco} recommandations complètes`);
console.log(`Écart numérique maximal : ${maxDiff.toExponential(2)} (tolérance ${TOL})`);
if (errors.length) {
  console.log(`ÉCHEC : ${errors.length} différence(s)\n  ` + errors.slice(0, 15).join("\n  "));
  process.exit(1);
}
console.log("RÉUSSI : le site en ligne donne exactement les mêmes réponses que l'API Python.");
