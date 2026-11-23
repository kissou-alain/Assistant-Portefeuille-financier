"use client";
// Champs de formulaire du diagnostic
import clsx from "clsx";
import type { ReactNode } from "react";

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-text">{label}</span>
      {hint && <span className="block text-xs text-muted">{hint}</span>}
      <div className="mt-1.5">{children}</div>
    </label>
  );
}

const inputCls = "w-full rounded-xl border border-border bg-surface px-3 py-2.5 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20";

export function NumberInput({ value, onChange, min = 0, max, step = 1, suffix }: {
  value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; suffix?: string;
}) {
  return (
    <div className="relative">
      <input type="number" inputMode="decimal" className={clsx(inputCls, "tabular", suffix && "pr-14")} value={Number.isFinite(value) ? value : ""}
        min={min} max={max} step={step} onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))} />
      {suffix && <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-sm text-muted">{suffix}</span>}
    </div>
  );
}

export function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <select className={inputCls} value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

export function Range({ value, onChange, min, max, labels }: { value: number; onChange: (v: number) => void; min: number; max: number; labels?: [string, string] }) {
  return (
    <div>
      <div className="flex items-center gap-3">
        <input type="range" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} className="flex-1 accent-[var(--primary)]" />
        <span className="w-8 text-right text-sm font-semibold tabular">{value}</span>
      </div>
      {labels && <div className="mt-1 flex justify-between text-xs text-muted"><span>{labels[0]}</span><span>{labels[1]}</span></div>}
    </div>
  );
}

export function Toggle<T extends string | boolean>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <div className="inline-flex rounded-xl border border-border bg-surface-2 p-1">
      {options.map((o) => (
        <button key={String(o.value)} type="button" onClick={() => onChange(o.value)}
          className={clsx("rounded-lg px-4 py-1.5 text-sm transition", value === o.value ? "bg-surface font-medium text-text shadow-sm" : "text-muted")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ChoiceList({ options, value, onChange }: { options: string[]; value?: number; onChange: (i: number) => void }) {
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {options.map((o, i) => (
        <button key={o} type="button" onClick={() => onChange(i)}
          className={clsx("rounded-xl border px-4 py-3 text-left text-sm transition",
            value === i ? "border-primary bg-primary-soft font-medium text-text" : "border-border bg-surface text-text hover:border-primary/50")}>
          {o}
        </button>
      ))}
    </div>
  );
}
