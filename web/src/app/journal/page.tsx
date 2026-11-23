"use client";
import clsx from "clsx";
import { Bell, Brain, Cpu, LineChart, Scale, ShieldCheck, UserRound, Waves } from "lucide-react";
import { Badge, Button, Card, ErrorBox, Notice, PageSkeleton, PageTitle } from "@/components/ui";
import { SimpleMarkdown } from "@/components/markdown";
import { dateFr } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { DEFAULT_FORM, toApi, useSavedForm } from "@/lib/store";

const AGENTS: Record<string, { icon: any; color: string }> = {
  "Agent Profil": { icon: UserRound, color: "bg-sky-500" },
  "Agent Régime": { icon: Waves, color: "bg-emerald-500" },
  "Agent Alerte": { icon: Bell, color: "bg-red-500" },
  "Agent Prévision": { icon: LineChart, color: "bg-violet-500" },
  "Agent Risque": { icon: ShieldCheck, color: "bg-amber-500" },
  "Agent Allocation": { icon: Scale, color: "bg-teal-600" },
  "Agent Explication": { icon: Brain, color: "bg-indigo-500" },
  Orchestrateur: { icon: Cpu, color: "bg-slate-500" },
};

const FLOW = ["Agent Profil", "Agent Régime", "Agent Alerte", "Agent Prévision", "Agent Risque", "Agent Allocation", "Agent Explication"];

export default function Journal() {
  const { form, ready } = useSavedForm();
  const f = form ?? DEFAULT_FORM;
  const res = useApi(ready ? "/api/recommandation" : null, ready ? { ...toApi(f), profil_force: f.profil_force ?? null, montant: f.montant ?? 100_000 } : undefined);

  if (!ready || (!res.data && !res.error)) return <PageSkeleton />;
  if (res.error) return <ErrorBox message={res.error} retry={res.reload} />;
  const d = res.data;
  // Les messages dont le résumé est vide ou égal à la clé correspondent à des données brutes publiées sur le tableau noir
  const events = d.journal.map((e: any) => ({ ...e, brut: !e.summary || e.summary === e.key }));

  return (
    <div className="space-y-6">
      <PageTitle title="Dialogue des agents" subtitle={`Chronologie des échanges sur le tableau noir pendant la recommandation du ${dateFr(d.date)}`}
        action={<Button href="/portefeuille" variant="secondary">Voir le portefeuille</Button>} />
      {!form && <Notice>Exemple avec un client type : faites votre diagnostic pour voir le dialogue correspondant à votre situation.</Notice>}

      <Card title="Architecture" subtitle="Chaque agent lit et publie sur un tableau noir partagé ; l'orchestrateur fixe l'ordre et gère la négociation Allocation ⇄ Risque">
        <div className="flex flex-wrap items-center gap-2">
          {FLOW.map((a, i) => {
            const A = AGENTS[a];
            return (
              <div key={a} className="flex items-center gap-2">
                <span className={clsx("flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium text-white", A.color)}><A.icon className="h-3.5 w-3.5" />{a.replace("Agent ", "")}</span>
                {i < FLOW.length - 1 && <span className="text-muted">→</span>}
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-sm text-muted">Si l'agent Risque détecte un dépassement des limites du profil, l'orchestrateur réduit la cible de volatilité de 10 % et relance l'agent Allocation (jusqu'à 3 tours).</p>
      </Card>

      <Card title="Journal des messages" subtitle={`${events.filter((e: any) => !e.brut).length} messages · profil ${d.profil} · régime ${d.regime}`}>
        <ol className="relative ml-4 border-l border-border">
          {events.map((e: any, i: number) => {
            const A = AGENTS[e.sender] ?? AGENTS.Orchestrateur;
            return (
              <li key={i} className={clsx("mb-4 ml-6", e.brut && "opacity-60")}>
                <span className={clsx("absolute -left-3.5 flex h-7 w-7 items-center justify-center rounded-full text-white ring-4 ring-surface", A.color)}><A.icon className="h-3.5 w-3.5" /></span>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-semibold">{e.sender}</span>
                  <Badge tone="x">{e.key}</Badge>
                  <span className="text-xs text-muted">{e.timestamp}</span>
                </div>
                <p className="mt-0.5 text-sm text-text">{e.brut ? <span className="text-muted">données publiées sur le tableau noir</span> : e.summary}</p>
              </li>
            );
          })}
        </ol>
      </Card>

      <Card title="Rapport final de l'agent Explication"><SimpleMarkdown text={d.explication} /></Card>
    </div>
  );
}
