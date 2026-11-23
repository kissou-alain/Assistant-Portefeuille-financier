"use client";
import { useState } from "react";
import { Area, CartesianGrid, Cell, ComposedChart, Legend, Line, Pie, PieChart, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { Badge, Button, Card, ErrorBox, Notice, PageSkeleton, PageTitle, Stat } from "@/components/ui";
import { NumberInput } from "@/components/fields";
import { SimpleMarkdown } from "@/components/markdown";
import { dateFr, num, pct } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { toApi, useSavedForm } from "@/lib/store";

const CLASS_COLORS: Record<string, string> = {
  Actions: "var(--chart-2)", Immobilier: "var(--chart-3)", Obligations: "var(--chart-1)",
  "Monétaire": "var(--chart-5)", "Matières premières": "var(--chart-4)", Crypto: "var(--chart-6)",
};
const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 12 };

export default function Portefeuille() {
  const { form, ready, update } = useSavedForm();
  const [montant, setMontant] = useState<number | null>(null);
  const body = form ? { ...toApi(form), profil_force: form.profil_force ?? null, montant: form.montant ?? 100_000 } : undefined;
  const res = useApi(form ? "/api/recommandation" : null, body);

  if (!ready) return <PageSkeleton />;
  if (!form) return (
    <div className="mx-auto max-w-xl text-center"><PageTitle title="Mon portefeuille" />
      <Card><p className="text-muted">Faites d'abord votre diagnostic pour obtenir une recommandation.</p>
        <Button href="/diagnostic" className="mt-4">Commencer mon diagnostic</Button></Card></div>
  );
  if (res.error) return <ErrorBox message={res.error} retry={res.reload} />;
  if (!res.data) return <PageSkeleton />;
  const d = res.data;
  const cur = form.devise;
  const classes = Object.entries(d.classes as Record<string, number>).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
  const proj = d.projection.points.map((p: any) => ({ ...p, b90: [p.q5, p.q95], b50: [p.q25, p.q75] }));
  const last = d.projection.points[d.projection.points.length - 1];

  return (
    <div className="space-y-6">
      <PageTitle title="Portefeuille recommandé"
        subtitle={<>Au {dateFr(d.date)} · profil <Badge>{d.profil}</Badge> · régime de marché <Badge>{d.regime}</Badge></>}
        action={<Button href="/journal" variant="secondary">Voir le dialogue des agents</Button>} />

      {d.alerte_marche?.alerte && <Notice tone="warning"><b>Agent Alerte :</b> {d.alerte_marche.message}</Notice>}
      <Notice>{d.devise_note}{!d.prevision_active && " Les prévisions de rendement sont affichées à titre indicatif : elles ne battent pas une prévision naïve de façon significative et n'influencent donc pas l'allocation."}</Notice>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Volatilité annuelle attendue" value={pct(d.risque.volatilite)} hint={`Cible du profil : ${pct(d.risque.cible_volatilite, 0)}`} />
        <Stat label="VaR 95 % à 1 mois" value={pct(d.risque.var95_1mois)} hint="Perte mensuelle dépassée 1 mois sur 20" />
        <Stat label="CVaR 95 % (1 jour)" value={pct(d.risque.cvar95_jour)} hint="Perte moyenne des 5 % pires jours" />
        <Stat label="Pire baisse sur 1 an" value={pct(d.risque.perte_max_fenetre)} hint="Si ce portefeuille avait été détenu" tone="bad" />
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card title="Répartition par classe d'actifs" className="lg:col-span-2">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={classes} dataKey="value" nameKey="name" innerRadius="55%" outerRadius="85%" paddingAngle={2}>
                  {classes.map((c) => <Cell key={c.name} fill={CLASS_COLORS[c.name] ?? "var(--chart-5)"} />)}
                </Pie>
                <Tooltip formatter={(v: any) => pct(v)} contentStyle={tooltipStyle} />
                <Legend formatter={(v: string) => <span className="text-sm text-text">{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Lignes du portefeuille" subtitle={`${d.lignes.length} positions`} className="lg:col-span-3">
          <div className="max-h-80 overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface text-left text-xs uppercase text-muted">
                <tr><th className="py-2">Actif</th><th>Classe</th><th className="text-right">Poids</th>
                  <th className="text-right">Sur 1 an</th><th className="text-right" title="Écart de rendement prévu à 1 mois vs la moyenne des actifs">Prévision*</th></tr>
              </thead>
              <tbody>
                {d.lignes.map((l: any) => (
                  <tr key={l.ticker} className="border-t border-border">
                    <td className="py-2"><span className="font-semibold">{l.ticker}</span><span className="block text-xs text-muted">{l.nom}</span></td>
                    <td className="text-muted">{l.classe}</td>
                    <td className="text-right font-medium tabular">
                      {pct(l.poids)}
                      <div className="ml-auto mt-1 h-1 w-20 rounded bg-surface-2"><div className="h-1 rounded bg-primary" style={{ width: `${Math.min(100, l.poids * 400)}%` }} /></div>
                    </td>
                    <td className={`text-right tabular ${l.rendement_1an >= 0 ? "text-success" : "text-danger"}`}>{pct(l.rendement_1an)}</td>
                    <td className="text-right tabular text-muted">{l.prevision === null ? "–" : `${l.prevision > 0 ? "+" : ""}${pct(l.prevision)}`}<span className="block text-xs">rang {l.rang}/25</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">* Indicatif : prévision de l'agent Prévision, non utilisée dans l'allocation (garde-fou statistique).</p>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Frontière efficiente" subtitle={d.frontiere.note}>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid stroke="var(--border)" />
                <XAxis type="number" dataKey="volatilite" name="Volatilité" tickFormatter={(v) => pct(v, 0)} tick={{ fontSize: 12, fill: "var(--muted)" }}
                  label={{ value: "Volatilité annuelle", position: "insideBottom", offset: -10, fontSize: 12, fill: "var(--muted)" }} domain={[0, "auto"]} />
                <YAxis type="number" dataKey="rendement" name="Rendement" tickFormatter={(v) => pct(v, 0)} tick={{ fontSize: 12, fill: "var(--muted)" }} />
                <ZAxis range={[40, 40]} />
                <Tooltip formatter={(v: any) => pct(v)} contentStyle={tooltipStyle} labelFormatter={() => ""} />
                <Legend verticalAlign="top" height={30} />
                <Scatter name="Frontière" data={d.frontiere.points} fill="var(--chart-1)" line={{ stroke: "var(--chart-1)", strokeWidth: 2 }} shape={() => <g />} />
                <Scatter name="Actifs" data={d.frontiere.actifs} fill="var(--chart-5)" />
                <Scatter name="Votre portefeuille" data={[d.frontiere.portefeuille]} fill="var(--stress)" shape="star" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Projection de la valeur" subtitle="Fourchettes illustratives (modèle log-normal), hors frais et hors effet de change"
          action={<div className="w-40"><NumberInput value={montant ?? form.montant ?? 100_000} onChange={setMontant} step={1000} suffix={cur} /></div>}>
          {montant !== null && montant !== form.montant && (
            <div className="mb-2"><Button variant="secondary" onClick={() => update({ ...form, montant })}>Recalculer pour {montant.toLocaleString("fr-FR")} {cur}</Button></div>
          )}
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={proj} margin={{ top: 10, right: 20, bottom: 0, left: 10 }}>
                <CartesianGrid stroke="var(--border)" />
                <XAxis dataKey="annee" tickFormatter={(v) => `${v} an${v > 1 ? "s" : ""}`} tick={{ fontSize: 12, fill: "var(--muted)" }} />
                <YAxis tickFormatter={(v) => (v / 1000).toFixed(0) + " k"} tick={{ fontSize: 12, fill: "var(--muted)" }} />
                <Tooltip formatter={(v: any) => Array.isArray(v) ? v.map((x) => Math.round(x).toLocaleString("fr-FR")).join(" – ") : Math.round(v).toLocaleString("fr-FR")} contentStyle={tooltipStyle} />
                <Area dataKey="b90" name="90 % des scénarios" stroke="none" fill="var(--chart-1)" fillOpacity={0.15} />
                <Area dataKey="b50" name="50 % des scénarios" stroke="none" fill="var(--chart-1)" fillOpacity={0.3} />
                <Line dataKey="q50" name="Médiane" stroke="var(--chart-1)" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-sm text-muted">
            À 10 ans : médiane <b className="text-text">{Math.round(last.q50).toLocaleString("fr-FR")} {cur}</b>, entre {Math.round(last.q5).toLocaleString("fr-FR")} et {Math.round(last.q95).toLocaleString("fr-FR")} {cur} dans 90 % des scénarios
            (rendement espéré {pct(d.projection.rendement_espere)} par an, volatilité {num(d.projection.volatilite * 100, 1)} %).
          </p>
        </Card>
      </div>

      <Card title="L'explication des agents"><SimpleMarkdown text={d.explication} /></Card>
    </div>
  );
}
