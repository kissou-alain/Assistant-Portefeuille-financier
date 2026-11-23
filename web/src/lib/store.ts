"use client";
// Réponses du diagnostic, conservées dans le navigateur (confort de l'utilisateur uniquement)
import { useEffect, useState } from "react";

export type ClientForm = {
  age: number; enfants: number; revenu: number; actifs: number; dettes: number;
  devise: string; taux_change: number; connaissance: number; quiz: number; sexe: string;
  etudes: string; en_couple: boolean; situation: string; proprietaire: boolean; a_epargne: boolean;
  reponses: Record<string, number>; profil_force?: string | null; montant?: number;
};

export const DEFAULT_FORM: ClientForm = {
  age: 40, enfants: 1, revenu: 700_000, actifs: 3_000_000, dettes: 1_000_000, devise: "MAD", taux_change: 10,
  connaissance: 5, quiz: 2, sexe: "Homme", etudes: "Diplôme universitaire", en_couple: true, situation: "Salarié",
  proprietaire: true, a_epargne: true, reponses: {}, profil_force: null, montant: 100_000,
};

const KEY = "assistant-portefeuille:diagnostic";

export function loadForm(): ClientForm | null {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? { ...DEFAULT_FORM, ...JSON.parse(raw) } : null;
  } catch {
    return null;
  }
}

export function saveForm(f: ClientForm) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(f));
  } catch {}
}

/** Formulaire sauvegardé ; `ready` passe à true une fois le navigateur lu (évite un écart serveur/client). */
export function useSavedForm() {
  const [form, setForm] = useState<ClientForm | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => { setForm(loadForm()); setReady(true); }, []);
  const update = (f: ClientForm) => { saveForm(f); setForm(f); };
  return { form, ready, update };
}

/** Corps de requête pour l'API : le taux de change n'est envoyé que hors USD */
export function toApi(f: ClientForm) {
  const { montant, profil_force, ...rest } = f;
  return { ...rest, taux_change: f.devise === "USD" ? 1 : f.taux_change };
}
