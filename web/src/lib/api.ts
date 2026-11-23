// Client de l'API FastAPI (port 8000 par défaut)
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {}

export async function api<T = any>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError("L'API ne répond pas. Vérifiez qu'elle est lancée (start.ps1).");
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
