"use client";
// Composants d'interface réutilisables (cartes, boutons, badges, indicateurs, états de chargement)
import clsx from "clsx";
import Link from "next/link";
import { AlertTriangle, Info } from "lucide-react";
import type { ReactNode } from "react";

export function Card({ title, subtitle, action, children, className }: {
  title?: ReactNode; subtitle?: ReactNode; action?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={clsx("rounded-2xl border border-border bg-surface p-5 shadow-sm", className)}>
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold text-text">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Button({ children, onClick, href, variant = "primary", disabled, type = "button", className }: {
  children: ReactNode; onClick?: () => void; href?: string; variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean; type?: "button" | "submit"; className?: string;
}) {
  const cls = clsx(
    "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition disabled:opacity-50",
    variant === "primary" && "bg-primary text-white hover:opacity-90 dark:text-slate-900",
    variant === "secondary" && "border border-border bg-surface text-text hover:bg-surface-2",
    variant === "ghost" && "text-muted hover:text-text hover:bg-surface-2",
    className,
  );
  if (href) return <Link href={href} className={cls}>{children}</Link>;
  return <button type={type} onClick={onClick} disabled={disabled} className={cls}>{children}</button>;
}

const BADGE = {
  Prudent: "bg-sky-100 text-sky-800 dark:bg-sky-900/50 dark:text-sky-200",
  "Équilibré": "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200",
  Dynamique: "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-200",
  Calme: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-200",
  Neutre: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-200",
  Stress: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200",
} as Record<string, string>;

export function Badge({ children, tone }: { children: ReactNode; tone?: string }) {
  return (
    <span className={clsx("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
      BADGE[tone ?? String(children)] ?? "bg-surface-2 text-muted")}>
      {children}
    </span>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: "good" | "bad" }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">{label}</p>
      <p className={clsx("mt-1 text-2xl font-semibold tabular", tone === "good" && "text-success", tone === "bad" && "text-danger")}>{value}</p>
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-xl bg-surface-2", className)} />;
}

export function PageSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-64" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-24" />)}</div>
      <Skeleton className="h-72" />
    </div>
  );
}

export function ErrorBox({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-2xl border border-danger/40 bg-danger/10 p-4 text-sm">
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-danger" />
      <div className="flex-1">
        <p className="font-medium text-text">Impossible d'afficher cette page</p>
        <p className="text-muted">{message}</p>
      </div>
      {retry && <Button variant="secondary" onClick={retry}>Réessayer</Button>}
    </div>
  );
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warning" }) {
  return (
    <div className={clsx("flex items-start gap-2 rounded-xl border p-3 text-sm",
      tone === "warning" ? "border-warning/40 bg-warning/10" : "border-border bg-surface-2")}>
      {tone === "warning" ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
        : <Info className="mt-0.5 h-4 w-4 shrink-0 text-muted" />}
      <div className="text-text">{children}</div>
    </div>
  );
}

export function PageTitle({ title, subtitle, action }: { title: string; subtitle?: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-text sm:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1 text-muted">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}
