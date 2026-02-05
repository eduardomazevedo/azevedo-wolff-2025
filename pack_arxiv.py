#!/usr/bin/env python3
"""
pack_arxiv: Build an arxiv-ready zip of the paper.

Scans tex/ for:
  - \\includegraphics{...} to find which figures are actually used
  - \\input{...} to find which .tex files are needed

Creates a staging directory with:
  - All needed .tex files (manuscript, si, and everything they \\input)
  - custom.sty, bibliography.bib (and .bbl if present)
  - tex/tables/ contents
  - Only the figure files that appear in the paper (copies real files, not symlinks)

Then zips the staging dir as arxiv_submission.zip.

Usage (from project root):
  python pack_arxiv.py
  python pack_arxiv.py --out my_arxiv.zip
  python pack_arxiv.py --keep-staging   # leave arxiv_staging/ for inspection

Figures are read from tex/figures/ (symlinks are followed; only referenced
figures are copied). \\input paths outside tex/ (e.g. ../output/...) are
not copied; a note is printed at the end if any are found.
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create arxiv submission zip for the paper.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root (default: directory containing this script)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output zip path (default: <root>/arxiv_submission.zip)",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep staging directory after creating zip",
    )
    args = parser.parse_args()

    root = args.root
    tex_dir = root / "tex"
    if not tex_dir.is_dir():
        raise SystemExit(f"tex directory not found: {tex_dir}")

    # 1) Find all .tex and .sty files under tex/
    tex_files = list(tex_dir.rglob("*.tex")) + list(tex_dir.glob("*.sty"))
    # 2) Parse \input{path} and \includegraphics[...]{path} / \includegraphics{path}
    input_re = re.compile(r"\\input\s*\{([^}]+)\}")
    # Optional [width=...] then {path}; path may omit extension
    graphics_re = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")

    def normalize_input(path: str) -> Path:
        p = path.strip()
        if not p.endswith(".tex"):
            p = p + ".tex"
        return Path(p)

    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    # Collect all \input targets reachable from manuscript.tex only (si.tex not included in arxiv bundle)
    roots = [tex_dir / "manuscript.tex"]
    needed_tex: set[Path] = set()
    to_visit = [r.relative_to(tex_dir) for r in roots if r.exists()]
    while to_visit:
        rel = to_visit.pop()
        if rel in needed_tex:
            continue
        needed_tex.add(rel)
        full = tex_dir / rel
        if not full.exists():
            continue
        text = read_file(full)
        for m in input_re.finditer(text):
            inp = normalize_input(m.group(1))
            # Resolve relative to current file's directory (under tex/)
            base = rel.parent if rel.parent != Path(".") else Path(".")
            full = (tex_dir / base / inp).resolve()
            try:
                resolved = full.relative_to(tex_dir)
            except ValueError:
                continue  # \input outside tex/ (e.g. ../output/...)
            if resolved not in needed_tex:
                to_visit.append(resolved)

    # Collect all figure paths from every .tex and .sty in tex/
    figure_paths: set[str] = set()
    for f in tex_files:
        text = read_file(f)
        for m in graphics_re.finditer(text):
            path = m.group(1).strip()
            if path.startswith("figures/") or "figures/" in path:
                figure_paths.add(path)

    # Normalize figure paths: ensure we store as relative path under tex/
    figures_to_copy: list[Path] = []
    for fp in sorted(figure_paths):
        # Paths are relative to tex/ (e.g. figures/log-poisson/pp_stacked.pdf)
        if not fp.startswith("figures/"):
            fp = "figures/" + fp.split("figures/")[-1] if "figures/" in fp else fp
        src = tex_dir / fp
        if src.exists():
            figures_to_copy.append(Path(fp))
        else:
            # Try with .pdf if no extension
            if not Path(fp).suffix:
                src = tex_dir / (fp + ".pdf")
                if src.exists():
                    figures_to_copy.append(Path(fp + ".pdf"))

    # Staging directory
    stage = root / "arxiv_staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    stage_tex = stage / "tex"
    stage_tex.mkdir(parents=True)

    # Copy needed .tex files (preserving relative paths)
    for rel in needed_tex:
        src = tex_dir / rel
        if src.exists():
            dst = stage_tex / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Copy custom.sty and bibliography.bib
    for name in ("custom.sty", "bibliography.bib"):
        src = tex_dir / name
        if src.exists():
            shutil.copy2(src, stage_tex / name)
    bbl = tex_dir / "bibliography.bbl"
    if bbl.exists():
        shutil.copy2(bbl, stage_tex / "bibliography.bbl")

    # Copy tables/
    tables_src = tex_dir / "tables"
    if tables_src.is_dir():
        shutil.copytree(tables_src, stage_tex / "tables", dirs_exist_ok=True)

    # Copy only used figures (follow symlinks so zip contains real files)
    for fp in figures_to_copy:
        src = tex_dir / fp
        if not src.exists():
            continue
        dst = stage_tex / fp
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=True)

    # Zip
    out_zip = args.out or (root / "arxiv_submission.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in stage.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(stage)
                zf.write(f, arcname)

    print(f"Created {out_zip}")
    print(f"  Tex files: {len(needed_tex)}")
    print(f"  Figures:   {len(figures_to_copy)}")
    if not args.keep_staging:
        shutil.rmtree(stage)
        print("  Staging directory removed.")
    else:
        print(f"  Staging left at: {stage}")

    # Warn about external inputs (../output/ etc.) that are not copied
    all_tex_text = ""
    for rel in needed_tex:
        all_tex_text += read_file(tex_dir / rel)
    external_inputs = input_re.findall(all_tex_text)
    external = [x.strip() for x in external_inputs if x.strip().startswith("../") or "/output/" in x]
    if external:
        print("\n  Note: these \\input paths point outside tex/ and were not copied:")
        for e in sorted(set(external)):
            print(f"    - {e}")


if __name__ == "__main__":
    main()
