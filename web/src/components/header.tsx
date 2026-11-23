"use client";
import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Compass, Menu, Moon, Sun, X } from "lucide-react";

const LINKS = [
  { href: "/diagnostic", label: "Diagnostic" },
  { href: "/profil", label: "Mon profil" },
  { href: "/portefeuille", label: "Portefeuille" },
  { href: "/marche", label: "Marché" },
  { href: "/backtest", label: "Backtest" },
  { href: "/laboratoire", label: "Laboratoire" },
  { href: "/journal", label: "Agents" },
  { href: "/donnees", label: "Données" },
];

export function Header() {
  const path = usePathname();
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => setMounted(true), []);
  useEffect(() => setOpen(false), [path]);

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-surface/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-accent">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-white dark:text-slate-900">
            <Compass className="h-5 w-5" />
          </span>
          <span className="hidden sm:inline">Assistant Portefeuille</span>
        </Link>
        <nav className="ml-4 hidden flex-1 items-center gap-1 lg:flex">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href}
              className={clsx("rounded-lg px-3 py-2 text-sm transition",
                path.startsWith(l.href) ? "bg-primary-soft font-medium text-primary" : "text-muted hover:bg-surface-2 hover:text-text")}>
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1">
          <button aria-label="Changer de thème" onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
            className="rounded-lg p-2 text-muted hover:bg-surface-2 hover:text-text">
            {mounted && resolvedTheme === "dark" ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>
          <button aria-label="Menu" onClick={() => setOpen(!open)} className="rounded-lg p-2 text-muted hover:bg-surface-2 lg:hidden">
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>
      {open && (
        <nav className="border-t border-border bg-surface px-4 py-2 lg:hidden">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href}
              className={clsx("block rounded-lg px-3 py-2 text-sm", path.startsWith(l.href) ? "bg-primary-soft text-primary" : "text-text")}>
              {l.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
