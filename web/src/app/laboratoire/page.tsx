"use client";
import clsx from "clsx";
import { useState } from "react";
import { X } from "lucide-react";
import { Badge, Card, ErrorBox, Notice, PageSkeleton, PageTitle, Stat } from "@/components/ui";
import { SimpleMarkdown } from "@/components/markdown";
import { assetUrl, num, pct } from "@/lib/api";
import { useApi } from "@/lib/use-api";

const TABS = ["Agent Profil", "Agent Prévision", "Agent Régime", "Figures", "Fiches des modèles"] as const;

/** Tableau générique : première colonne = nom, nombres arrondis à 3 décimales */
function DataTable({ rows, highlight, hide = [] }: { rows?: any[]; highlight?: string; hide?: string[] }) {
  if (!rows?.length) return <p className="text-sm text-muted">Tableau indisponible (exécutez le notebook).</p>;
  const cols = Object.keys(rows[0]).filter((c) => !hide.includes(c));
  const fmt = (v: any) => (typeof v === "number" ? num(v, Math.abs(v) >= 100 ? 0 : 3) : v ?? "–");
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-muted"><tr>{cols.map((c) => <th key={c} className="py-2 pr-3 font-medium">{c === "nom" ? "" : c}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i} className={clsx("border-t border-border", highlight && String(r.nom).includes(highlight) && "bg-primary-soft/40 font-medium",
            String(r.nom).startsWith("Baseline") && "text-muted")}>
            {cols.map((c) => <td key={c} className={clsx("py-1.5 pr-3", c !== "nom" && "tabular")}>{fmt(r[c])}</td>)}
          </tr>))}
        </tbody>
      </table>
    </div>
  );
}

export default function Laboratoire() {
  const lab = useApi("/api/modeles");
  const [tab, setTab] = useState<(typeof TABS)[number]>("Agent Profil");
  const [zoom, setZoom] = useState<any>(null);
  if (lab.error) return <ErrorBox message={lab.error} retry={lab.reload} />;
  if (!lab.data) return <PageSkeleton />;
  const { meta, tables: t, figures, fiches } = lab.data;

  return (
    <div className="space-y-6">
      <PageTitle title="Laboratoire des modèles" subtitle="Méthodologie, comparaisons et tests statistiques – résultats produits par le notebook" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Modèle Profil" value={<span className="text-lg">{meta.modele_profil}</span>} hint={`Seuil de décision ${pct(meta.seuil_profil, 0)}`} />
        <Stat label="Modèle Prévision" value={<span className="text-lg">{meta.modele_prevision}</span>}
          hint={meta.prevision_active ? "Activé dans l'allocation" : "Non activé (garde-fou de Diebold-Mariano)"} />
        <Stat label="Réglages de l'allocation" value={<span className="text-lg">γ = {meta.penalite_rotation}</span>} hint={`Réduction du risque en stress : ${meta.reduction_risque_regime}`} />
        <Stat label="Données" value={<span className="text-lg">{meta.date_donnees}</span>} hint={meta.mode_rapide ? "⚠ résultats du mode rapide" : "Exécution complète du notebook"} />
      </div>

      <div className="flex flex-wrap gap-1 rounded-xl border border-border bg-surface-2 p-1">
        {TABS.map((x) => <button key={x} onClick={() => setTab(x)} className={clsx("rounded-lg px-3 py-1.5 text-sm", x === tab ? "bg-surface font-medium shadow-sm" : "text-muted")}>{x}</button>)}
      </div>

      {tab === "Agent Profil" && (
        <div className="space-y-4">
          <Card title="Comparaison des modèles (validation)" subtitle="Classification binaire Prudent / Non prudent – critère AUC ROC ; découpage par ménage">
            <DataTable rows={t.profil_validation} highlight={meta.modele_profil} />
          </Card>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Classes déséquilibrées" subtitle="Poids de classes, SMOTE et ADASYN (appliqués à l'entraînement seulement)"><DataTable rows={t.profil_desequilibre} /></Card>
            <Card title="Test et validation externe (SCF 2019)" subtitle="Évaluation unique ; puis application sans ré-entraînement à une autre vague de l'enquête">
              <DataTable rows={t.profil_test} /><div className="mt-4"><DataTable rows={t.profil_externe} /></div>
            </Card>
          </div>
        </div>
      )}

      {tab === "Agent Prévision" && (
        <div className="space-y-4">
          <Notice>Régression du rendement excédentaire à 21 jours (fenêtre glissante : L = 252, H = 21, S = 5). Un modèle n'est déclaré meilleur que s'il bat la référence au <b>test de Diebold-Mariano</b> (DM &gt; 0, p &lt; 0,05).</Notice>
          <Card title="Prévisions hors pli 2013-2021 et tests de Diebold-Mariano" subtitle="Critère de choix du modèle ; environ 400 dates, sans toucher au test">
            <DataTable rows={t.prevision_diebold_mariano} highlight={meta.modele_prevision} />
          </Card>
          <Card title="Trois méthodes d'évaluation" subtitle="Hold-out temporel, validation croisée expanding window et rolling window (mêmes hyperparamètres)">
            <DataTable rows={t.prevision_trois_evaluations} hide={["nb plis"]} />
          </Card>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Comparaison sur la validation 2020-2021"><DataTable rows={t.prevision_validation} hide={["famille", "MAE", "RMSE", "Spread Q5-Q1 brut (annualisé)"]} /></Card>
            <Card title="Test 2022 → (évaluation unique)"><DataTable rows={t.prevision_test} hide={["RMSE", "MAE", "Spread Q5-Q1 brut (annualisé)"]} />
              <div className="mt-4"><p className="mb-2 text-sm font-medium">Apport des probabilités de régime (GMM) comme variables</p><DataTable rows={t.prevision_apport_gamma} /></div>
            </Card>
          </div>
        </div>
      )}

      {tab === "Agent Régime" && (
        <div className="space-y-4">
          <Card title="Les quatre familles de clustering" subtitle="Silhouette ↑, Davies-Bouldin ↓, Calinski-Harabasz ↑ ; accord avec le GMM (indice de Rand ajusté)">
            <DataTable rows={t.regime_clustering} highlight="GMM" />
          </Card>
          <Card title="Validation économique" subtitle="Rendement et volatilité du S&P 500 le mois suivant, selon le régime détecté">
            <DataTable rows={t.regime_validation} />
          </Card>
        </div>
      )}

      {tab === "Figures" && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {figures.map((f: any) => (
            <button key={f.id} onClick={() => setZoom(f)} className="overflow-hidden rounded-2xl border border-border bg-surface text-left transition hover:shadow-md">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={assetUrl(f.url)} alt={f.titre} className="h-48 w-full bg-white object-contain" />
              <p className="p-3 text-sm font-medium">{f.titre}</p>
            </button>
          ))}
        </div>
      )}

      {tab === "Fiches des modèles" && (
        <Card title="Fiches explicatives des modèles" subtitle="Intuition, fonction de coût, hyperparamètres et raison du choix de chaque modèle">
          {fiches ? <SimpleMarkdown text={fiches} /> : <p className="text-sm text-muted">Les fiches (docs/fiches_modeles.md) seront ajoutées avec les livrables.</p>}
        </Card>
      )}

      {zoom && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={() => setZoom(null)}>
          <div className="max-h-full max-w-6xl overflow-auto rounded-2xl bg-white p-3" onClick={(e) => e.stopPropagation()}>
            <div className="mb-2 flex items-center justify-between"><Badge tone="x">{zoom.titre}</Badge>
              <button onClick={() => setZoom(null)} className="rounded-lg p-1 text-slate-600 hover:bg-slate-100"><X className="h-5 w-5" /></button></div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={assetUrl(zoom.url)} alt={zoom.titre} className="h-auto w-full" />
          </div>
        </div>
      )}
    </div>
  );
}
