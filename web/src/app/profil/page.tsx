"use client";
import clsx from "clsx";
import { ArrowRight, ClipboardList, Cpu, UserCheck } from "lucide-react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Button, Card, ErrorBox, Notice, PageSkeleton, PageTitle } from "@/components/ui";
import { pct } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { toApi, useSavedForm } from "@/lib/store";

const LEVELS = ["Prudent", "Équilibré", "Dynamique"];

function Gauge({ profil, declare, capacite }: { profil: string; declare: string; capacite: string }) {
  const pos = (p: string) => LEVELS.indexOf(p);
  return (
    <div>
      <div className="grid grid-cols-3 gap-1.5">
        {LEVELS.map((l, i) => (
          <div key={l} className={clsx("rounded-lg py-3 text-center text-sm font-semibold transition",
            i === pos(profil) ? ["bg-sky-500 text-white", "bg-amber-500 text-white", "bg-rose-500 text-white"][i] : "bg-surface-2 text-muted")}>
            {l}
          </div>
        ))}
      </div>
      <div className="mt-3 grid grid-cols-3 text-center text-xs text-muted">
        {LEVELS.map((l) => (
          <div key={l} className="space-y-0.5">
            {declare === l && <p>▲ déclaré</p>}
            {capacite === l && <p>▲ capacité (modèle)</p>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Profil() {
  const { form, ready, update } = useSavedForm();
  const res = useApi(form ? "/api/profil" : null, form ? toApi(form) : undefined);

  if (!ready) return <PageSkeleton />;
  if (!form) return (
    <div className="mx-auto max-w-xl text-center">
      <PageTitle title="Mon profil" />
      <Card><p className="text-muted">Vous n'avez pas encore fait votre diagnostic.</p>
        <Button href="/diagnostic" className="mt-4">Commencer mon diagnostic</Button></Card>
    </div>
  );
  if (res.error) return <ErrorBox message={res.error} retry={res.reload} />;
  if (!res.data) return <PageSkeleton />;
  const d = res.data;
  const profilFinal = form.profil_force || d.profil;
  const contrib = [...d.explication.contributions].reverse();

  return (
    <div className="space-y-6">
      <PageTitle title="Mon profil d'investisseur" subtitle="Le profil retenu est le plus prudent entre ce que vous déclarez et ce que votre situation permet."
        action={<Button href="/portefeuille">Voir mon portefeuille <ArrowRight className="h-4 w-4" /></Button>} />

      <Card title={<span className="flex items-center gap-2">Profil retenu <Badge>{profilFinal}</Badge>
        {form.profil_force && <Badge tone="x">validé par le conseiller</Badge>}</span>}>
        <Gauge profil={profilFinal} declare={d.declare} capacite={d.capacite === "aucun plafond" ? "" : d.capacite} />
        {d.alerte && <div className="mt-4"><Notice tone="warning"><b>Incohérence détectée.</b> {d.alerte}</Notice></div>}
        {d.questionnaire.plafonds.map((p: string) => <div key={p} className="mt-2"><Notice>{p}</Notice></div>)}
      </Card>

      <div className="grid gap-4 md:grid-cols-3">
        <Card title={<span className="flex items-center gap-2"><ClipboardList className="h-4 w-4 text-primary" /> Tolérance déclarée</span>} subtitle="Questionnaire noté (7 questions)">
          <p className="text-3xl font-bold tabular">{d.questionnaire.score}<span className="text-base text-muted"> / {d.questionnaire.score_max}</span></p>
          <div className="mt-2 h-2 rounded-full bg-surface-2"><div className="h-2 rounded-full bg-primary" style={{ width: `${(100 * d.questionnaire.score) / d.questionnaire.score_max}%` }} /></div>
          <p className="mt-3 text-sm">→ <Badge>{d.declare}</Badge></p>
          <p className="mt-2 text-xs text-muted">Seuils : moins de 8 points = Prudent, de 8 à 14 = Équilibré, 15 et plus = Dynamique.</p>
        </Card>
        <Card title={<span className="flex items-center gap-2"><Cpu className="h-4 w-4 text-primary" /> Capacité estimée</span>} subtitle={`Modèle : ${d.modele}`}>
          <p className="text-3xl font-bold tabular">{pct(d.proba_prudent, 0)}</p>
          <p className="text-sm text-muted">probabilité de ressembler aux ménages prudents</p>
          <div className="relative mt-3 h-2 rounded-full bg-surface-2">
            <div className="h-2 rounded-full bg-sky-500" style={{ width: `${100 * d.proba_prudent}%` }} />
            <div className="absolute -top-1 h-4 w-0.5 bg-text" style={{ left: `${100 * d.seuil}%` }} title="seuil" />
          </div>
          <p className="mt-3 text-sm">→ {d.capacite === "aucun plafond" ? <span className="text-muted">aucun plafond imposé</span> : <Badge>{d.capacite}</Badge>}</p>
          <p className="mt-2 text-xs text-muted">Seuil de décision {pct(d.seuil, 0)}, choisi pour repérer au moins 80 % des clients prudents.</p>
        </Card>
        <Card title={<span className="flex items-center gap-2"><UserCheck className="h-4 w-4 text-primary" /> Validation du conseiller</span>} subtitle="Le conseiller peut confirmer ou corriger le profil">
          <div className="space-y-2">
            {[null, ...LEVELS].map((p) => (
              <button key={String(p)} onClick={() => update({ ...form, profil_force: p })}
                className={clsx("w-full rounded-xl border px-3 py-2 text-left text-sm",
                  (form.profil_force ?? null) === p ? "border-primary bg-primary-soft font-medium" : "border-border hover:bg-surface-2")}>
                {p ? `Imposer le profil ${p}` : `Retenir la proposition des agents (${d.profil})`}
              </button>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Pourquoi ce profil ?" subtitle="Contribution de chaque caractéristique à l'estimation de votre capacité (valeurs SHAP, explication locale)">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={contrib} layout="vertical" margin={{ left: 10, right: 30 }}>
              <XAxis type="number" tick={{ fontSize: 12, fill: "var(--muted)" }} />
              <YAxis type="category" dataKey="libellé" width={230} tick={{ fontSize: 12, fill: "var(--text)" }} />
              <ReferenceLine x={0} stroke="var(--muted)" />
              <Tooltip formatter={(v: any) => Number(v).toFixed(3)} contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12 }} />
              <Bar dataKey="contribution" radius={[4, 4, 4, 4]}>
                {contrib.map((c: any) => <Cell key={c["libellé"]} fill={c.contribution > 0 ? "var(--stress)" : "var(--calme)"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-sm text-muted">
          <span className="font-medium text-stress">Rouge</span> : la caractéristique vous rapproche des ménages prudents ;{" "}
          <span className="font-medium text-calme">vert</span> : elle vous en éloigne. Les valeurs sont exprimées en log-odds
          (échelle logistique) ; leur somme, ajoutée à la valeur de base, donne la probabilité affichée ci-dessus.
        </p>
      </Card>
    </div>
  );
}
