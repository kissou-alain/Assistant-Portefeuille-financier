// Affichage du Markdown simple produit par l'agent Explication (titres ###, **gras**, _italique_, listes « - »)
import type { ReactNode } from "react";

function inline(s: string): ReactNode[] {
  return s.split(/(\*\*[^*]+\*\*|_[^_]+_)/g).filter(Boolean).map((p, i) =>
    p.startsWith("**") ? <strong key={i}>{p.slice(2, -2)}</strong>
      : p.startsWith("_") && p.endsWith("_") ? <em key={i} className="text-muted">{p.slice(1, -1)}</em> : p);
}

export function SimpleMarkdown({ text }: { text: string }) {
  const out: ReactNode[] = [];
  let list: string[] = [];
  const flush = () => {
    if (list.length) out.push(<ul key={out.length} className="my-2 list-disc space-y-1 pl-5">{list.map((l, i) => <li key={i}>{inline(l)}</li>)}</ul>);
    list = [];
  };
  text.split("\n").forEach((line) => {
    if (line.startsWith("- ")) { list.push(line.slice(2)); return; }
    flush();
    if (line.startsWith("### ")) out.push(<h3 key={out.length} className="mb-2 text-base font-semibold">{inline(line.slice(4))}</h3>);
    else if (line.trim()) out.push(<p key={out.length} className="my-1.5">{inline(line)}</p>);
  });
  flush();
  return <div className="text-sm leading-relaxed text-text">{out}</div>;
}
