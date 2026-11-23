// Client de l'API FastAPI (port 8000 par défaut) – ou mode « en ligne » (site statique, sans serveur Python)
import { offlineApi } from "./offline";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {}

/** Vrai sur le site publié (Vercel) : pas d'API Python, réponses lues dans /donnees et profil calculé dans le navigateur */
export const isOnline = () =>
  !process.env.NEXT_PUBLIC_API_URL && typeof window !== "undefined"
  && !["localhost", "127.0.0.1"].includes(window.location.hostname);

/** Adresse d'un fichier servi par l'API en local (figures), par le site lui-même en ligne */
export const assetUrl = (url: string) => (isOnline() ? url : `${API_URL}${url}`);

export async function api<T = any>(path: string, body?: unknown): Promise<T> {
  if (isOnline()) {
    try {
      return await offlineApi(path, body);
    } catch (e) {
      throw new ApiError(e instanceof Error ? e.message : String(e));
    }
  }
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(API_URL.includes("localhost")
      ? "L'API ne répond pas. Vérifiez qu'elle est lancée (start.ps1)."
      : "Le serveur de calcul se réveille (hébergement gratuit mis en veille après 15 minutes d'inactivité). Patientez environ une minute puis cliquez sur « Réessayer ».");
  }
  if (!res.ok) {
    let detail = `Erreur ${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new ApiError(detail);
  }
  return res.json();
}

// Formatage (français)
export const pct = (v?: number | null, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d })} %`;
export const num = (v?: number | null, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : v.toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });
export const money = (v?: number | null, cur = "USD") =>
  v === null || v === undefined ? "–" : v.toLocaleString("fr-FR", { maximumFractionDigits: 0 }) + " " + cur;
export const dateFr = (s?: string) => (s ? new Date(s).toLocaleDateString("fr-FR") : "–");
