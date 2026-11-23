"use client";
import clsx from "clsx";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import { Button, Card, ErrorBox, Notice, PageSkeleton, PageTitle } from "@/components/ui";
import { ChoiceList, Field, NumberInput, Range, Select, Toggle } from "@/components/fields";
import { useApi } from "@/lib/use-api";
import { ClientForm, DEFAULT_FORM, loadForm, saveForm } from "@/lib/store";

const STEPS = ["Situation financière", "Tolérance au risque", "Récapitulatif"];

export default function Diagnostic() {
  const router = useRouter();
  const q = useApi("/api/questionnaire");
  const [step, setStep] = useState(0);
  const [f, setF] = useState<ClientForm>(DEFAULT_FORM);
  useEffect(() => { const s = loadForm(); if (s) setF(s); }, []);
  const set = <K extends keyof ClientForm>(k: K, v: ClientForm[K]) => setF((p) => ({ ...p, [k]: v }));

  if (q.error) return <ErrorBox message={q.error} retry={q.reload} />;
  if (!q.data) return <PageSkeleton />;
  const devise = q.data.devises.find((d: any) => d.code === f.devise) ?? q.data.devises[0];
  const sym = devise.symbol;
  const nbRep = q.data.questions.filter((x: any) => f.reponses[x.cle] !== undefined).length;
  const complet = nbRep === q.data.questions.length;

  const finish = () => { saveForm({ ...f, profil_force: null }); router.push("/profil"); };

  return (
    <div className="mx-auto max-w-3xl">
      <PageTitle title="Votre diagnostic" subtitle="Environ 3 minutes. Vos réponses restent dans votre navigateur." />
      {/* Barre de progression */}
      <ol className="mb-6 grid grid-cols-3 gap-2">
        {STEPS.map((s, i) => (
          <li key={s}>
            <div className={clsx("h-1.5 rounded-full", i <= step ? "bg-primary" : "bg-surface-2")} />
            <p className={clsx("mt-2 text-xs sm:text-sm", i === step ? "font-semibold text-text" : "text-muted")}>{i + 1}. {s}</p>
          </li>
        ))}
      </ol>

      {step === 0 && (
        <Card title="Votre situation" subtitle="Ces informations servent à estimer votre capacité à supporter des pertes.">
          <div className="grid gap-5 sm:grid-cols-2">
            <Field label="Devise des montants">
              <Select value={f.devise} onChange={(v) => setF((p) => ({ ...p, devise: v, taux_change: q.data.devises.find((d: any) => d.code === v)?.per_usd ?? 1 }))}
                options={q.data.devises.map((d: any) => ({ value: d.code, label: d.label }))} />
            </Field>
            {f.devise !== "USD" ? (
              <Field label={`Taux de change : 1 USD = … ${f.devise}`} hint="Taux indicatif, à ajuster au cours du jour">
                <NumberInput value={f.taux_change} onChange={(v) => set("taux_change", v)} min={0.01} step={0.01} />
              </Field>
            ) : <div />}
            <Field label="Âge"><NumberInput value={f.age} onChange={(v) => set("age", v)} min={18} max={100} suffix="ans" /></Field>
            <Field label="Enfants à charge"><NumberInput value={f.enfants} onChange={(v) => set("enfants", v)} max={15} /></Field>
            <Field label="Revenu annuel du foyer"><NumberInput value={f.revenu} onChange={(v) => set("revenu", v)} step={1000} suffix={sym} /></Field>
            <Field label="Total de vos actifs" hint="Épargne, placements, immobilier compris"><NumberInput value={f.actifs} onChange={(v) => set("actifs", v)} step={1000} suffix={sym} /></Field>
            <Field label="Total de vos dettes" hint="Crédit immobilier, consommation…"><NumberInput value={f.dettes} onChange={(v) => set("dettes", v)} step={1000} suffix={sym} /></Field>
            <Field label="Niveau d'études">
              <Select value={f.etudes} onChange={(v) => set("etudes", v)} options={q.data.etudes.map((e: string) => ({ value: e, label: e }))} />
            </Field>
            <Field label="Situation professionnelle">
              <Select value={f.situation} onChange={(v) => set("situation", v)} options={q.data.situations.map((e: string) => ({ value: e, label: e }))} />
            </Field>
            <Field label="Connaissance des questions financières"><Range value={f.connaissance} onChange={(v) => set("connaissance", v)} min={0} max={10} labels={["aucune", "excellente"]} /></Field>
            <Field label="Bonnes réponses à un quiz de culture financière" hint="Sur 3 questions (taux d'intérêt, inflation, diversification)">
              <Range value={f.quiz} onChange={(v) => set("quiz", v)} min={0} max={3} />
            </Field>
            <Field label="Sexe de la personne de référence">
              <Toggle value={f.sexe} onChange={(v) => set("sexe", v)} options={[{ value: "Homme", label: "Homme" }, { value: "Femme", label: "Femme" }]} />
            </Field>
            <Field label="En couple"><Toggle value={f.en_couple} onChange={(v) => set("en_couple", v)} options={[{ value: true, label: "Oui" }, { value: false, label: "Non" }]} /></Field>
            <Field label="Propriétaire de votre logement"><Toggle value={f.proprietaire} onChange={(v) => set("proprietaire", v)} options={[{ value: true, label: "Oui" }, { value: false, label: "Non" }]} /></Field>
            <Field label="Avez-vous épargné l'an dernier ?"><Toggle value={f.a_epargne} onChange={(v) => set("a_epargne", v)} options={[{ value: true, label: "Oui" }, { value: false, label: "Non" }]} /></Field>
          </div>
          <div className="mt-5">
            <Notice>Le modèle a été entraîné sur des ménages américains (montants en dollars) : vos montants sont convertis en USD avec le taux indiqué.</Notice>
          </div>
        </Card>
      )}

      {step === 1 && (
        <div className="space-y-4">
          {q.data.questions.map((x: any, i: number) => (
            <Card key={x.cle} title={`${i + 1}. ${x.intitule}`}>
              <ChoiceList options={x.options} value={f.reponses[x.cle]} onChange={(v) => set("reponses", { ...f.reponses, [x.cle]: v })} />
            </Card>
          ))}
        </div>
      )}

      {step === 2 && (
        <Card title="Récapitulatif" subtitle="Vérifiez vos réponses avant de lancer l'analyse.">
          <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
            {[["Âge", `${f.age} ans`], ["Enfants", f.enfants], ["Revenu annuel", `${f.revenu.toLocaleString("fr-FR")} ${sym}`],
              ["Actifs", `${f.actifs.toLocaleString("fr-FR")} ${sym}`], ["Dettes", `${f.dettes.toLocaleString("fr-FR")} ${sym}`],
              ["Études", f.etudes], ["Situation", f.situation], ["Connaissances", `${f.connaissance}/10`]].map(([k, v]) => (
              <div key={String(k)} className="flex justify-between border-b border-border py-1.5"><dt className="text-muted">{k}</dt><dd className="font-medium">{v}</dd></div>
            ))}
          </dl>
          <p className="mt-4 text-sm">Questionnaire : <b>{nbRep}</b> réponse(s) sur {q.data.questions.length}.</p>
          {!complet && <div className="mt-3"><Notice tone="warning">Les questions sans réponse seront comptées comme la réponse la plus prudente.</Notice></div>}
        </Card>
      )}

      <div className="mt-6 flex justify-between">
        <Button variant="secondary" onClick={() => setStep(step - 1)} disabled={step === 0}><ArrowLeft className="h-4 w-4" /> Précédent</Button>
        {step < 2 ? <Button onClick={() => { saveForm(f); setStep(step + 1); window.scrollTo(0, 0); }}>Suivant <ArrowRight className="h-4 w-4" /></Button>
          : <Button onClick={finish}><Check className="h-4 w-4" /> Voir mon profil</Button>}
      </div>
    </div>
  );
}
