"use client";
import clsx from "clsx";
import { useMemo, useState } from "react";
import { CartesianGrid, Cell, ComposedChart, Line, LineChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Card, ErrorBox, Notice, PageSkeleton, PageTitle, Skeleton, Stat } from "@/components/ui";
import { dateFr, num, pct } from "@/lib/api";
import { useApi } from "@/lib/use-api";

const REG_COLOR: Record<string, string> = { Calme: "var(--calme)", Neutre: "var(--neutre)", Stress: "var(--stress)" };
const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 };
const shortDate = (s: string) => new Date(s).toLocaleDateString("fr-FR", { month: "short", year: "2-digit" });

export default function Marche() {
  const m = useApi("/api/marche");
  const [ticker, setTicker] = useState("SPY");
  const [tri, setTri] = useState<"rendement_1an" | "rendement_1m" | "volatilite_3m">("rendement_1an");
  const prix = useApi(`/api/prix/${encodeURIComponent(ticker)}`);
  const actifs = useMemo(() => (m.data?.actifs ?? []).slice().sort((a: any, b: any) => b[tri] - a[tri]), [m.data, tri]);

  if (m.error) return <ErrorBox message={m.error} retry={m.reload} />;
  if (!m.data) return <PageSkeleton />;
  const d = m.data;
  const ind = d.indicateurs;

  return (
    <div className="space-y-6">
      <PageTitle title="État du marché" subtitle={`Données au ${dateFr(d.date)} – agents Régime (mélange gaussien) et Alerte (détection d'anomalies)`} />

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Régime actuel" subtitle="Probabilité de chaque régime (sortie du mélange gaussien)">
          <div className="mb-4 flex items-center gap-2"><Badge>{d.regime}</Badge>
            {d.alerte_du_jour ? <Badge tone="Stress">Journée anormale</Badge> : <span className="text-xs text-muted">pas d'anomalie aujourd'hui ({d.votes_du_jour}/4 détecteurs)</span>}
          </div>
          {(["Calme", "Neutre", "Stress"] as const).map((r) => (
            <div key={r} className="mb-3">
              <div className="flex justify-between text-sm"><span>{r}</span><span className="tabular">{pct(d.proba[r], 0)}</span></div>
              <div className="mt-1 h-2 rounded-full bg-surface-2"><div className="h-2 rounded-full" style={{ width: `${100 * d.proba[r]}%`, background: REG_COLOR[r] }} /></div>
            </div>
          ))}
          <p className="mt-2 text-xs text-muted">Un régime « Stress » annonce une volatilité plus forte le mois suivant, pas forcément une baisse.</p>
        </Card>
        <Card title="Régimes détectés sur le S&P 500" subtitle="3 dernières années" className="lg:col-span-2">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={d.historique} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="var(--border)" />
                <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={40} tick={{ fontSize: 12, fill: "var(--muted)" }} />
                <YAxis domain={["auto", "auto"]} tick={{ fontSize: 12, fill: "var(--muted)" }} />
                <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => dateFr(String(v))} formatter={(v: any, n: any) => [num(v, 2), n === "spy" ? "S&P 500 (SPY)" : n]} />
                <Line dataKey="spy" stroke="var(--muted)" strokeWidth={1} dot={false} isAnimationActive={false} />
                <Scatter dataKey="spy" isAnimationActive={false}>
                  {d.historique.map((p: any, i: number) => <Cell key={i} fill={REG_COLOR[p.regime]} />)}
                </Scatter>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex gap-4 text-xs">{Object.entries(REG_COLOR).map(([k, c]) => <span key={k} className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full" style={{ background: c }} />{k}</span>)}</div>
        </Card>
      </div>

      <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="VIX" value={num(ind.VIX, 1)} hint="Volatilité implicite" />
        <Stat label="S&P 500 (1 mois)" value={pct(ind["S&P 500 sur 1 mois"])} tone={ind["S&P 500 sur 1 mois"] >= 0 ? "good" : "bad"} />
        <Stat label="Volatilité S&P 500" value={pct(ind["Volatilité S&P 500 (1 mois)"])} hint="Réalisée, sur 1 mois" />
        <Stat label="Écart au plus haut" value={pct(ind["Écart au plus haut sur 1 an"])} hint="S&P 500, sur 1 an" />
        <Stat label="Pente 10 ans – 2 ans" value={`${num(ind["Pente 10 ans – 2 ans (%)"], 2)} pt`} hint="Courbe des taux US" />
        <Stat label="Spread de crédit" value={`${num(ind["Spread de crédit Baa (%)"], 2)} pt`} hint="Baa – Trésor 10 ans" />
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card title="Actifs de l'univers" subtitle="Cliquez sur un actif pour afficher son cours" className="lg:col-span-3"
          action={<div className="flex gap-1 text-xs">{([["rendement_1m", "1 mois"], ["rendement_1an", "1 an"], ["volatilite_3m", "Volatilité"]] as const).map(([k, l]) => (
            <button key={k} onClick={() => setTri(k)} className={clsx("rounded-lg px-2 py-1", tri === k ? "bg-primary-soft text-primary" : "text-muted hover:bg-surface-2")}>{l}</button>))}</div>}>
          <div className="max-h-96 overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface text-left text-xs uppercase text-muted"><tr><th className="py-2">Actif</th><th>Classe</th><th className="text-right">1 mois</th><th className="text-right">1 an</th><th className="text-right">Vol. 3 mois</th></tr></thead>
              <tbody>
                {actifs.map((a: any) => (
                  <tr key={a.ticker} onClick={() => setTicker(a.ticker)} className={clsx("cursor-pointer border-t border-border hover:bg-surface-2", a.ticker === ticker && "bg-primary-soft/50")}>
                    <td className="py-2"><b>{a.ticker}</b> <span className="text-xs text-muted">{a.nom}</span></td>
                    <td className="text-muted">{a.classe}</td>
                    <td className={clsx("text-right tabular", a.rendement_1m >= 0 ? "text-success" : "text-danger")}>{pct(a.rendement_1m)}</td>
                    <td className={clsx("text-right tabular", a.rendement_1an >= 0 ? "text-success" : "text-danger")}>{pct(a.rendement_1an)}</td>
                    <td className="text-right tabular">{pct(a.volatilite_3m)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <div className="space-y-4 lg:col-span-2">
          <Card title={prix.data ? `${prix.data.ticker} – ${prix.data.nom}` : ticker} subtitle="Cours ajusté, 3 ans (USD)">
            <div className="h-56">
              {prix.data ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={prix.data.points} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <XAxis dataKey="date" tickFormatter={shortDate} minTickGap={40} tick={{ fontSize: 11, fill: "var(--muted)" }} />
                    <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11, fill: "var(--muted)" }} width={45} />
                    <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => dateFr(String(v))} formatter={(v: any) => [num(v, 2), "Cours"]} />
                    <Line dataKey="prix" stroke="var(--primary)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : <Skeleton className="h-full" />}
            </div>
          </Card>
          <Card title="Journées anormales récentes" subtitle="Au moins 3 détecteurs sur 4 d'accord (Agent Alerte)">
            {d.alertes.length === 0 ? <p className="text-sm text-muted">Aucune alerte sur la période.</p> : (
              <ul className="max-h-40 space-y-1 overflow-auto text-sm">
                {d.alertes.slice().reverse().map((a: any) => <li key={a.date} className="flex justify-between"><span>{dateFr(a.date)}</span><span className="text-muted">{a.votes}/4 détecteurs</span></li>)}
              </ul>
            )}
            <div className="mt-3"><Notice>Les seuils ont été appris sur 2011-2019 ; le marché étant plus agité depuis 2020, les alertes sont plus fréquentes (dérive des données).</Notice></div>
          </Card>
        </div>
      </div>
    </div>
  );
}
