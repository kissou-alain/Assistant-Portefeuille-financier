"use client";
import { useState } from "react";
import { CheckCircle2, Database, Loader2, RefreshCw, RotateCcw, XCircle } from "lucide-react";
import { Button, Card, ErrorBox, Notice, PageSkeleton, PageTitle, Stat } from "@/components/ui";
import { api, dateFr } from "@/lib/api";
import { useApi } from "@/lib/use-api";

type Msg = { ok: boolean; text: string } | null;

export default function Donnees() {
  const s = useApi("/api/donnees");
  const [busy, setBusy] = useState<"maj" | "restaurer" | null>(null);
  const [msg, setMsg] = useState<Msg>(null);

  const run = async (kind: "maj" | "restaurer") => {
    if (kind === "restaurer" && !window.confirm("Revenir aux données sauvegardées avant la dernière mise à jour ?")) return;
    setBusy(kind);
    setMsg(null);
    try {
      const r = await api(`/api/donnees/${kind}`, {});
      setMsg({ ok: true, text: r.message });
      s.reload();
    } catch (e: any) {
      setMsg({ ok: false, text: e.message });
    } finally {
      setBusy(null);
    }
  };

  if (s.error) return <ErrorBox message={s.error} retry={s.reload} />;
  if (!s.data) return <PageSkeleton />;
  const d = s.data;

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <PageTitle title="Données de marché" subtitle="Cours Yahoo Finance (25 actifs) et séries macroéconomiques FRED" />
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="Dernière date de marché" value={dateFr(d.date_donnees)} hint="Utilisée pour les recommandations" />
        <Stat label="Modèles entraînés sur des données au" value={dateFr(d.modeles_entraines_le)} hint="Ré-entraînement : relancer le notebook" />
        <Stat label="Sauvegarde" value={d.sauvegarde_disponible ? "Disponible" : "Aucune"} hint="Version précédant la dernière mise à jour" />
      </div>

      <Card title="Mettre à jour les données" subtitle="Téléchargement des derniers jours uniquement – 30 secondes au maximum">
        <ul className="mb-4 space-y-1.5 text-sm text-text">
          <li>• Seuls les derniers jours de cotation sont téléchargés, puis raccordés à l'historique.</li>
          <li>• Toutes les variables sont recalculées et <b>contrôlées avant écriture</b> (actifs manquants, dates plus anciennes, résultats vides).</li>
          <li>• En cas de problème (réseau lent, limitation de Yahoo Finance), <b>rien n'est modifié</b>.</li>
          <li>• Les données actuelles sont sauvegardées avant toute écriture : on peut toujours revenir en arrière.</li>
          <li>• Les modèles ne sont pas ré-entraînés : ils s'appliquent aux nouvelles données.</li>
        </ul>
        <div className="flex flex-wrap gap-3">
          <Button onClick={() => run("maj")} disabled={busy !== null}>
            {busy === "maj" ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            {busy === "maj" ? "Mise à jour en cours…" : "Mettre à jour les données"}
          </Button>
          <Button variant="secondary" onClick={() => run("restaurer")} disabled={busy !== null || !d.sauvegarde_disponible}>
            {busy === "restaurer" ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
            Revenir aux données précédentes
          </Button>
        </div>
        {msg && (
          <div className={`mt-4 flex items-start gap-2 rounded-xl border p-3 text-sm ${msg.ok ? "border-success/40 bg-success/10" : "border-danger/40 bg-danger/10"}`}>
            {msg.ok ? <CheckCircle2 className="mt-0.5 h-4 w-4 text-success" /> : <XCircle className="mt-0.5 h-4 w-4 text-danger" />}
            <span>{msg.text}</span>
          </div>
        )}
      </Card>

      <Card title={<span className="flex items-center gap-2"><Database className="h-4 w-4 text-primary" /> Sources (données ouvertes)</span>}>
        <ul className="space-y-2 text-sm">
          <li><b>Yahoo Finance</b> – cours ajustés quotidiens de 25 ETF et du Bitcoin depuis 2010.</li>
          <li><b>FRED</b> (Federal Reserve Bank of St. Louis) – taux, VIX, spread de crédit, inflation, chômage, production industrielle.</li>
          <li><b>Survey of Consumer Finances</b> 2022 et 2019 (Réserve fédérale) – ménages réels, pour l'agent Profil.</li>
        </ul>
        <div className="mt-4"><Notice>Les données Yahoo Finance sont réservées à un usage personnel et pédagogique.</Notice></div>
      </Card>
    </div>
  );
}
