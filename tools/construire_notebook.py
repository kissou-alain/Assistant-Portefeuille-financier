"""
Construit le notebook principal à partir des fichiers de sections `tools/sections/sNN_*.py`.

Chaque fichier de section définit une liste CELLS de cellules créées avec md("...") et code("...").
Le notebook produit est `notebooks/Projet_MultiAgent_Portefeuille_v4.ipynb` (les versions
précédentes du notebook ne sont jamais modifiées).

Usage :
    python tools/construire_notebook.py              # toutes les sections
    python tools/construire_notebook.py --jusqua 2   # sections 0 à 2 seulement (mise au point)
    python tools/construire_notebook.py --executer   # construit puis exécute (Restart & Run All)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "tools" / "sections"
OUT = ROOT / "notebooks" / "Projet_MultiAgent_Portefeuille_v4.ipynb"


def load_cells(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CELLS


def build(upto: int | None = None, skip: tuple = ()) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for f in sorted(SECTIONS.glob("s[0-9][0-9]_*.py")):
        num = int(f.name[1:3])
        if (upto is not None and num > upto) or num in skip:
            continue
        nb.cells.extend(load_cells(f))
    return nb


def keep_outputs(nb: nbformat.NotebookNode, executed: Path) -> int:
    """Recopie les sorties d'un notebook déjà exécuté dans `nb`, pour les cellules de code dont le
    code est strictement identique (utile quand seules des cellules de texte ont changé)."""
    old = [c for c in nbformat.read(executed, as_version=4).cells if c.cell_type == "code"]
    new = [c for c in nb.cells if c.cell_type == "code"]
    if len(old) != len(new):
        raise SystemExit(f"Nombre de cellules de code différent ({len(old)} → {len(new)}) : ré-exécuter le notebook.")
    for o, n in zip(old, new):
        if o.source != n.source:
            raise SystemExit("Une cellule de code a changé : ré-exécuter le notebook.\n" + n.source[:300])
        n.outputs, n.execution_count = o.outputs, o.execution_count
    return len(new)


def execute(path: Path, timeout: int = 5400) -> None:
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    try:
        ExecutePreprocessor(timeout=timeout, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(path.parent)}})
    except CellExecutionError as exc:
        nbformat.write(nb, path)          # on garde les sorties jusqu'à l'erreur, pour diagnostic
        msg = str(exc)
        print("ERREUR D'EXÉCUTION :", msg[-3000:])
        sys.exit(1)
    nbformat.write(nb, path)
    print(f"Exécution terminée en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.path.insert(0, str(SECTIONS))
    ap = argparse.ArgumentParser()
    ap.add_argument("--jusqua", type=int, default=None, help="dernière section incluse")
    ap.add_argument("--sauf", type=int, nargs="*", default=[], help="sections à exclure (mise au point)")
    ap.add_argument("--executer", action="store_true")
    ap.add_argument("--conserver-sorties", action="store_true",
                    help="reprend les sorties du notebook déjà exécuté (seul le texte a changé)")
    ap.add_argument("--sortie", type=str, default=str(OUT))
    a = ap.parse_args()
    out = Path(a.sortie)
    nb = build(a.jusqua, tuple(a.sauf))
    if a.conserver_sorties:
        print(f"{keep_outputs(nb, out)} cellules de code : sorties conservées")
    nbformat.write(nb, out)
    print(f"{len(nb.cells)} cellules écrites dans {out}")
    if a.executer:
        execute(out)
