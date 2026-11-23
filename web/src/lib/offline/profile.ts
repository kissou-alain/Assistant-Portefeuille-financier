// Agent Profil – version navigateur (mode en ligne).
// Reproduit EXACTEMENT le calcul Python (api/services.py → src/agents.py → pipeline scikit-learn) avec les
// paramètres appris exportés par tools/exporter_site.py (modele_profil.json). Aucun apprentissage ici :
// mêmes formules, mêmes nombres. L'égalité avec Python est vérifiée par tools/test_profil_parite.py.
// (Fichier sans import ni syntaxe TypeScript non effaçable : exécutable directement par Node pour le test.)

export type ProfileModel = any;
const ORDER = ["Prudent", "Équilibré", "Dynamique"];
const SITUATIONS: Record<string, number> = { "Salarié": 1, "Indépendant / chef d'entreprise": 2, "Retraité": 3, "Sans activité": 4 };
const ETUDES = ["Sans diplôme du secondaire", "Diplôme du secondaire", "Études supérieures courtes", "Diplôme universitaire"];

const mostPrudent = (...p: (string | null)[]) =>
  p.filter((x): x is string => !!x).sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b))[0];
const pct0 = (v: number) => `${Math.round(v * 100)}%`;

/** Formulaire → variables du SCF, montants convertis en dollars US (= services.build_client) */
export function buildClient(f: any, model: ProfileModel): Record<string, number> {
  const devise = f.devise ?? "USD";
  const perUsd = Number(f.taux_change || model.devises?.[devise]?.per_usd || 1);
  const income = Number(f.revenu ?? 0) / perUsd, assets = Number(f.actifs ?? 0) / perUsd, debt = Number(f.dettes ?? 0) / perUsd;
  const age = Math.trunc(Number(f.age ?? 40)), kids = Math.trunc(Number(f.enfants ?? 0));
  const married = f.en_couple ?? true;
  const situation = f.situation ?? "Salarié";
  const lifecl = age < 55 && !married && kids === 0 ? 1 : age < 55 && married && kids > 0 ? 2 : age < 55 && married ? 3
    : age < 55 ? 4 : situation !== "Retraité" ? 5 : 6;
  const saved = f.a_epargne ?? true;
  const c: Record<string, number> = {
    AGE: age, KIDS: kids, INCOME: income, NORMINC: income, ASSET: assets, DEBT: debt, NETWORTH: assets - debt,
    LEVRATIO: assets > 0 ? debt / assets : 0, DEBT2INC: income > 0 ? debt / income : 0,
    WAGEINC: situation === "Salarié" ? income : 0,
    KNOWL: Math.trunc(Number(f.connaissance ?? 5)), FINLIT: Math.trunc(Number(f.quiz ?? 2)),
    HHSEX: (f.sexe ?? "Homme") === "Homme" ? 1 : 2,
    EDCL: ETUDES.includes(f.etudes) ? ETUDES.indexOf(f.etudes) + 1 : 4,
    MARRIED: married ? 1 : 2, OCCAT1: SITUATIONS[situation] ?? 1, LIFECL: lifecl,
    HOUSECL: (f.proprietaire ?? true) ? 1 : 2, SAVED: saved ? 1 : 0, WSAVED: saved ? 3 : 1,
  };
  for (const [k, v] of Object.entries(f.reponses ?? {})) if (model.questionnaire.questions.includes(k)) c[k] = Math.trunc(Number(v));
  return c;
}

/** Questionnaire noté (= src/questionnaire.score_questionnaire) */
export function scoreQuestionnaire(c: Record<string, number>, model: ProfileModel) {
  const q = model.questionnaire;
  const detail: Record<string, number> = {};
  const caps: string[] = [];
  const capProfiles: string[] = [];
  let total = 0;
  for (const k of q.questions) {
    const v = c[k];
    const pts = v !== undefined && v !== null && v >= 0 && v <= 3 ? v : 0;
    detail[k] = pts;
    total += pts;
    const cap = q.plafonds.find((p: any) => p.question === k && p.reponse === pts);
    if (cap) { caps.push(`${q.libelles[k][pts]} → profil plafonné à ${cap.profil}`); capProfiles.push(cap.profil); }
  }
  let profile = total >= q.seuils["Dynamique"] ? "Dynamique" : total >= q.seuils["Équilibré"] ? "Équilibré" : "Prudent";
  if (capProfiles.length) profile = mostPrudent(profile, ...capProfiles);
  return { score: total, score_max: 3 * q.questions.length, profile, detail, plafonds: caps };
}

/** Prétraitement appris (ColumnTransformer) : vecteur de 115 valeurs dans l'ordre du modèle */
export function transform(c: Record<string, number>, m: ProfileModel): number[] {
  const x: number[] = [];
  const val = (k: string) => (c[k] === undefined || c[k] === null || Number.isNaN(c[k]) ? NaN : c[k]);
  m.montants.variables.forEach((k: string, i: number) => {          // médiane → log signé → standardisation
    let v = val(k);
    if (Number.isNaN(v)) v = m.montants.mediane[i];
    v = Math.sign(v) * Math.log1p(Math.abs(v));
    x.push((v - m.montants.moyenne[i]) / m.montants.ecart_type[i]);
  });
  m.numeriques.variables.forEach((k: string, i: number) => {        // médiane → écrêtage → standardisation
    let v = val(k);
    if (Number.isNaN(v)) v = m.numeriques.mediane[i];
    v = Math.min(Math.max(v, m.numeriques.bas[i]), m.numeriques.haut[i]);
    x.push((v - m.numeriques.moyenne[i]) / m.numeriques.ecart_type[i]);
  });
  for (const [k, spec] of Object.entries<any>(m.categorielles)) {    // modalité (ou imputation) → one-hot
    const v = val(k);
    const cat = Number.isNaN(v) ? spec.imputation : String(Math.trunc(v));
    const block = new Array(spec.size).fill(0);
    const j = spec.mapping[cat];
    if (j !== undefined && j >= 0) block[j] = 1;                     // modalité inconnue : que des zéros
    x.push(...block);
  }
  return x;
}

/** Agent Profil complet : même réponse que POST /api/profil */
export function profileFromForm(form: any, m: ProfileModel) {
  const c = buildClient(form, m);
  const x = transform(c, m);
  const z = m.constante + x.reduce((s, xi, j) => s + xi * m.coefficients[j], 0);
  const pNonPrudent = 1 / (1 + Math.exp(-z));
  const pPrud = 1 - pNonPrudent;
  const capacity = pPrud >= m.seuil ? "Prudent" : null;

  const hasAnswers = m.questionnaire.questions.some((k: string) => k in c);
  const q = scoreQuestionnaire(c, m);
  const declared = hasAnswers ? q.profile : "Équilibré";   // sans réponse : règle de repli (horizon et perte inconnus)
  const declTxt = hasAnswers ? `questionnaire ${q.score}/${q.score_max} → ${declared}` : `horizon nan ans, perte acceptable nan % → ${declared}`;
  const profil = capacity ? mostPrudent(declared, capacity) : declared;
  const alerte = capacity && ORDER.indexOf(declared) > ORDER.indexOf(capacity)
    ? `Incohérence : le client se déclare ${declared} mais sa situation financière ressemble à celle des ménages prudents. Profil ramené à ${profil} – à valider par le conseiller.`
    : null;
  const capTxt = capacity ? `profil financier proche des ménages prudents (p=${pPrud.toFixed(2)} ≥ seuil ${m.seuil.toFixed(2)})`
    : `pas de contrainte détectée (p(prudent)=${pPrud.toFixed(2)})`;

  // SHAP linéaire (interventionnel) : φ_j = β_j (x_j − E[x_j]) sur le log-odds de la classe 1 ; signe inversé pour « Prudent »
  const groups: Record<string, number> = {};
  x.forEach((xi, j) => {
    const g = m.groupes[j];
    groups[g] = (groups[g] ?? 0) - m.coefficients[j] * (xi - m.shap_moyenne_fond[j]);
  });
  const base = -(m.constante + m.coefficients.reduce((s: number, b: number, j: number) => s + b * m.shap_moyenne_fond[j], 0));
  const sorted = Object.entries(groups).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const contributions = sorted.slice(0, 8).map(([v, ct]) => ({ "libellé": m.libelles[v] ?? v, contribution: ct, effet: ct > 0 ? "augmente" : "diminue" }));
  if (sorted.length > 8) {
    const rest = sorted.slice(8).reduce((s, [, ct]) => s + ct, 0);
    contributions.push({ "libellé": `${sorted.length - 8} autres variables`, contribution: rest, effet: rest > 0 ? "augmente" : "diminue" });
  }
  return {
    profil, declare: declared, capacite: capacity ?? "aucun plafond", alerte, proba_prudent: pPrud, seuil: m.seuil,
    questionnaire: { score: q.score, score_max: q.score_max, plafonds: q.plafonds, detail: q.detail },
    explication: { valeur_de_base: base, contributions, unite: "log-odds de « Prudent » (positif = rend plus prudent)" },
    modele: m.modele,
    // éléments utilisés pour recomposer l'explication et le journal de la recommandation
    _texte: { declTxt, capTxt, decl: hasAnswers ? `questionnaire ${q.score}/${q.score_max} → ${declared}` : declared, probaTxt: pct0(pPrud) },
  };
}
