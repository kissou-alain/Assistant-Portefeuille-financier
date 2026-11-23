"use client";
import { ArrowRight, Bell, Brain, LineChart, Scale, ShieldCheck, UserRound, Waves } from "lucide-react";
import { Badge, Button, Card, ErrorBox, Skeleton, Stat } from "@/components/ui";
import { dateFr, pct } from "@/lib/api";
import { useApi } from "@/lib/use-api";

const AGENTS = [
  { icon: UserRound, nom: "Profil", texte: "Croise votre questionnaire et votre situation financière (régression logistique, SHAP)." },
  { icon: Waves, nom: "Régime", texte: "Lit l'état du marché : calme, neutre ou stress (mélange gaussien)." },
  { icon: Bell, nom: "Alerte", texte: "Signale les journées anormales (4 détecteurs d'anomalies)." },
  { icon: LineChart, nom: "Prévision", texte: "Évalue 13 modèles ; garde-fou statistique (test de Diebold-Mariano)." },
  { icon: ShieldCheck, nom: "Risque", texte: "Covariance de Ledoit-Wolf, volatilité, VaR, CVaR, limites du profil." },
  { icon: Scale, nom: "Allocation", texte: "Optimisation moyenne-variance avec a priori de Black-Litterman." },
  { icon: Brain, nom: "Explication", texte: "Justifie chaque recommandation et journalise les échanges entre agents." },
];

export default function Accueil() {
  const marche = useApi("/api/marche");
  const labo = useApi("/api/modeles");
  const meta = labo.data?.meta ?? {};

  return (
    <div className="space-y-10">
      <section className="grid items-center gap-8 lg:grid-cols-2">
        <div>
          <Badge tone="Calme">Projet de Machine Learning</Badge>
          <h1 className="mt-4 text-4xl font-bold leading-tight tracking-tight text-text sm:text-5xl">
            Un portefeuille adapté à <span className="text-primary">votre profil</span> et à <span className="text-accent">l'état du marché</span>.
          </h1>
          <p className="mt-4 text-lg text-muted">
            Sept agents spécialisés coopèrent pour estimer votre profil de risque, lire le marché, construire une
            allocation diversifiée sous contraintes et vous expliquer chaque choix.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button href="/diagnostic">Commencer mon diagnostic <ArrowRight className="h-4 w-4" /></Button>
            <Button href="/laboratoire" variant="secondary">Voir les modèles</Button>
          </div>
        </div>
        <Card title="Le marché aujourd'hui" subtitle={marche.data ? `Données au ${dateFr(marche.data.date)}` : "Chargement…"}>
          {marche.error ? <ErrorBox message={marche.error} retry={marche.reload} /> : !marche.data ? (
            <div className="space-y-3"><Skeleton className="h-10" /><Skeleton className="h-24" /></div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted">Régime détecté</span>
                <Badge>{marche.data.regime}</Badge>
                {marche.data.alerte_du_jour && <Badge tone="Stress">Alerte de marché</Badge>}
              </div>
              <div className="grid grid-cols-3 gap-3">
                {(["Calme", "Neutre", "Stress"] as const).map((r) => (
                  <div key={r} className="rounded-xl bg-surface-2 p-3 text-center">
                    <p className="text-xs text-muted">{r}</p>
                    <p className="text-lg font-semibold tabular">{pct(marche.data.proba[r], 0)}</p>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="flex justify-between"><span className="text-muted">VIX</span><span className="tabular">{marche.data.indicateurs.VIX?.toFixed(1)}</span></div>
                <div className="flex justify-between"><span className="text-muted">S&P 500 sur 1 mois</span><span className="tabular">{pct(marche.data.indicateurs["S&P 500 sur 1 mois"])}</span></div>
              </div>
            </div>
          )}
        </Card>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Actifs étudiés" value="25" hint="ETF actions, obligations, immobilier, matières premières, monétaire, Bitcoin" />
        <Stat label="Ménages réels" value="≈ 4 600" hint="Enquête de la Réserve fédérale (SCF 2022), validée sur 2019" />
        <Stat label="Modèles comparés" value="13 + 3" hint="13 modèles de prévision et 3 baselines, testés statistiquement" />
        <Stat label="Données de marché" value={meta.date_donnees ? dateFr(meta.date_donnees) : "–"} hint="Yahoo Finance et FRED, depuis 2010" />
      </section>

      <section>
        <h2 className="mb-4 text-xl font-semibold">Comment fonctionne l'assistant</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {AGENTS.map(({ icon: Icon, nom, texte }) => (
            <div key={nom} className="rounded-2xl border border-border bg-surface p-4">
              <Icon className="h-6 w-6 text-primary" />
              <p className="mt-3 font-semibold">Agent {nom}</p>
              <p className="mt-1 text-sm text-muted">{texte}</p>
            </div>
          ))}
          <div className="flex flex-col justify-between rounded-2xl bg-accent p-4 text-white dark:text-slate-900">
            <p className="font-semibold">Transparence avant tout</p>
            <p className="mt-1 text-sm opacity-90">Chaque résultat est comparé à des références simples et testé statistiquement.</p>
            <Button href="/backtest" variant="secondary" className="mt-3">Voir le backtest</Button>
          </div>
        </div>
      </section>
    </div>
  );
}
