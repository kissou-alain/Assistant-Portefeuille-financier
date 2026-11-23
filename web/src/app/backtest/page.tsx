"use client";
import clsx from "clsx";
import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Card, ErrorBox, Notice, PageSkeleton, PageTitle } from "@/components/ui";
import { dateFr, num, pct } from "@/lib/api";
import { useApi } from "@/lib/use-api";

const PROFILS = ["Prudent", "Équilibré", "Dynamique"];
const STRATS = [
  { s: "Multi-agent", color: "var(--chart-1)", width: 2.5 },
  { s: "Référence stratégique", color: "var(--chart-2)", width: 1.5 },
  { s: "1/N vol. cible", color: "var(--chart-4)", width: 1.5 },
  { s: "Parité de risque vol. cible", color: "var(--chart-3)", width: 1.5 },
  { s: "HRP vol. cible", color: "var(--chart-5)", width: 1.5 },
];
const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 };
const shortDate = (s: string) => new Date(s).toLocaleDateString("fr-FR", { month: "short", year: "2-digit" });

function Table({ rows, cols }: { rows: any[]; cols: { key: string; label: string; fmt?: (v: any) => string }[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-muted"><tr>{cols.map((c) => <th key={c.key} className="py-2 pr-3">{c.label}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i} className={clsx("border-t border-border", String(r.nom ?? "").includes("Multi-agent") && "bg-primary-soft/40 font-medium")}>
            {cols.map((c) => <td key={c.key} className="py-2 pr-3 tabular">{c.fmt ? c.fmt(r[c.key]) : r[c.key]}</td>)}
          </tr>))}
        </tbody>
      </table>
    </div>
  );
}

export default function Backtest() {
  const b = useApi("/api/backtest");
  const [profil, setProfil] = useState("Équilibré");
  if (b.error) return <ErrorBox message={b.error} retry={b.reload} />;
  if (!b.data) return <PageSkeleton />;
  const d = b.data;
  const keys = STRATS.map((x) => `${x.s} – ${profil}`).filter((k) => d.strategies.includes(k));
  const rc = (d.risque_comparable ?? []).filter((r: any) => r.nom.startsWith(profil + " / ")).map((r: any) => ({ ...r, nom: r.nom.split(" / ")[1] }));
  const sig = (d.significativite ?? []).filter((r: any) => r.nom.startsWith(profil + " :")).map((r: any) => ({ ...r, nom: r.nom.split(": ")[1].replace("Multi-agent vs ", "") }));
  const rl = (d.renforcement ?? []).filter((r: any) => r.nom.endsWith(profil));

  return (
    <div className="space-y-6">
      <PageTitle title="Backtest hors échantillon" subtitle="Période de test 2022 → aujourd'hui, jamais utilisée pour choisir les modèles ni les réglages"
        action={<div className="inline-flex rounded-xl border border-border bg-surface-2 p-1">{PROFILS.map((p) => (
          <button key={p} onClick={() => setProfil(p)} className={clsx("rounded-lg px-4 py-1.5 text-sm", p === profil ? "bg-surface font-medium shadow-sm" : "text-muted")}>{p}</button>))}</div>} />

      <Notice>Rebalancement mensuel, coûts de transaction de 10 points de base, covariance estimée sur l'année précédente. Les références « vol. cible » sont mélangées avec du monétaire pour ne pas dépasser le risque du profil (sans levier) : la parité de risque et HRP restent moins risquées que les profils Équilibré et Dynamique.</Notice>

      <Card title={<>Valeur d'un portefeuille de 1 – profil <Badge>{profil}</Badge></>}>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={d.series} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--border)" />
              <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={40} tick={{ fontSize: 12, fill: "var(--muted)" }} />
              <YAxis domain={["auto", "auto"]} tick={{ fontSize: 12, fill: "var(--muted)" }} tickFormatter={(v) => num(v, 2)} />
              <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => dateFr(String(v))} formatter={(v: any, n: any) => [num(v, 3), String(n).split(" – ")[0]]} />
              <Legend formatter={(v: string) => <span className="text-xs text-text">{v.split(" – ")[0]}</span>} />
              {STRATS.filter((x) => keys.includes(`${x.s} – ${profil}`)).map((x) => (
                <Line key={x.s} dataKey={`${x.s} – ${profil}`} stroke={x.color} strokeWidth={x.width} dot={false} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title="Pertes depuis le plus haut (drawdowns)">
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={d.drawdowns} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--border)" />
              <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={40} tick={{ fontSize: 12, fill: "var(--muted)" }} />
              <YAxis tickFormatter={(v) => pct(v, 0)} tick={{ fontSize: 12, fill: "var(--muted)" }} />
              <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => dateFr(String(v))} formatter={(v: any, n: any) => [pct(v), String(n).split(" – ")[0]]} />
              {STRATS.slice(0, 3).filter((x) => keys.includes(`${x.s} – ${profil}`)).map((x) => (
                <Line key={x.s} dataKey={`${x.s} – ${profil}`} stroke={x.color} strokeWidth={x.width} dot={false} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Performances à risque comparable" subtitle="Ratio de Sharpe = rendement excédentaire par unité de risque">
          <Table rows={rc} cols={[{ key: "nom", label: "Stratégie" }, { key: "Rendement annualisé", label: "Rendement", fmt: (v) => pct(v) },
            { key: "Volatilité annualisée", label: "Volatilité", fmt: (v) => pct(v) }, { key: "Sharpe", label: "Sharpe", fmt: (v) => num(v) },
            { key: "Max drawdown", label: "Perte max.", fmt: (v) => pct(v) }, { key: "Rotation annuelle", label: "Rotation", fmt: (v) => num(v) }]} />
        </Card>
        <Card title="Les écarts sont-ils significatifs ?" subtitle="Différence de Sharpe, intervalle de confiance à 95 % (bootstrap par blocs)">
          <Table rows={sig} cols={[{ key: "nom", label: "Multi-agent vs" }, { key: "diff. de Sharpe observée", label: "Écart", fmt: (v) => num(v) },
            { key: "IC 95 % bas", label: "IC bas", fmt: (v) => num(v) }, { key: "IC 95 % haut", label: "IC haut", fmt: (v) => num(v) },
            { key: "significatif (5 %)", label: "Significatif" }]} />
          <p className="mt-2 text-xs text-muted">Si l'intervalle contient 0, la différence n'est pas statistiquement établie.</p>
        </Card>
      </div>

      <Card title="Agent d'apprentissage par renforcement (PPO)" subtitle="Évalué une seule fois sur le test ; non retenu dans l'application (ne bat pas les références)">
        <Table rows={rl} cols={[{ key: "nom", label: "Stratégie" }, { key: "Rendement annualisé", label: "Rendement", fmt: (v) => pct(v) },
          { key: "Volatilité annualisée", label: "Volatilité", fmt: (v) => pct(v) }, { key: "Sharpe", label: "Sharpe", fmt: (v) => num(v) },
          { key: "Max drawdown", label: "Perte max.", fmt: (v) => pct(v) }]} />
      </Card>

      <Notice tone="warning">
        <b>Conclusion honnête :</b> sur 2022-2026, le système respecte le niveau de risque de chaque profil mais n'apporte pas de surperformance
        statistiquement établie face à des règles simples de même risque (et fait moins bien pour le profil Prudent). Sa valeur tient à
        l'adéquation au profil, au contrôle du risque et à l'explication des recommandations.
      </Notice>
    </div>
  );
}
