"use client";
// Petit hook de chargement : données, erreur, état de chargement et relance
import { useCallback, useEffect, useState } from "react";
import { api } from "./api";

export function useApi<T = any>(path: string | null, body?: unknown) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const key = JSON.stringify(body ?? null);

  const load = useCallback(() => {
    if (!path) return;
    setLoading(true);
    setError(null);
    api<T>(path, body)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, key]);

  useEffect(() => { load(); }, [load]);
  return { data, error, loading, reload: load };
}
