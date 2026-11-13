#!/usr/bin/env python
"""Build the customer deliverable: a clean ZIP with only what an end user needs.

    python tools/build_release.py                 # -> dist/TorusPlot-<version>.zip
    python tools/build_release.py --list          # print what would be included

The development repository also holds tests, validation fixtures, audit records, design
notes and developer tools. Those are NOT part of the product that is handed to customers.
This script defines the deliverable explicitly (an allowlist), so nothing else can slip in:

* included: the app, the `torusplot` package (with bundled OFL fonts and their license
  texts), launchers, pinned requirements, user documentation, synthetic examples, LICENSE and
  THIRD_PARTY_NOTICES.md
* excluded: `tests/`, `tools/`, `docs/dev/`, `docs/audit/`, reference notes, decision logs,
  CI configuration, developer requirements, anything untracked (e.g. `_reference/`, `.env`)

The archive is deterministic (sorted entries, fixed timestamps).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Top-level files and folders that make up the product.
INCLUDE = [
    "LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", ".env.example", "pyproject.toml",
    "requirements.txt", "requirements-lock.txt", "requirements-ai.txt",
    "run_torusplot.bat", "run_torusplot.sh", "cli.py",
    ".streamlit/config.toml", "app/", "torusplot/", "examples/",
    "docs/USER_GUIDE.md", "docs/STATS_METHODS.md", "docs/MCP.md", "docs/PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md",
    "docs/images/",
]
#: Never shipped, even if a rule above would match.
EXCLUDE = re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.ruff_cache)(/|$)|\.py[co]$|(^|/)\.env$|^_reference/|^tests/|^tools/")
_DEV_BLOCK = re.compile(r"<!-- dev-only:start -->.*?<!-- dev-only:end -->\n?", re.DOTALL)
_FIXED_DATE = (2026, 1, 1, 0, 0, 0)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def release_files() -> list[str]:
    """Tracked files that belong to the deliverable (allowlist minus exclusions)."""
    chosen = []
    for rel in tracked_files():
        if EXCLUDE.search(rel):
            continue
        if any(rel == inc or (inc.endswith("/") and rel.startswith(inc)) for inc in INCLUDE):
            chosen.append(rel)
    return sorted(chosen)


def release_readme() -> str:
    """README without the blocks that only make sense in the development repository."""
    return _DEV_BLOCK.sub("", (ROOT / "README.md").read_text(encoding="utf-8"))


def build(out: Path) -> Path:
    from torusplot import __version__

    files = release_files()
    missing = [inc for inc in INCLUDE if not any(f == inc or f.startswith(inc) for f in files)]
    if missing:
        raise SystemExit(f"Release is incomplete; not found among tracked files: {missing}")
    top = f"TorusPlot-{__version__}"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            data = release_readme().encode("utf-8") if rel == "README.md" else (ROOT / rel).read_bytes()
            info = zipfile.ZipInfo(f"{top}/{rel}", date_time=_FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if rel.endswith(".sh") else 0o644) << 16
            zf.writestr(info, data)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="Output ZIP path (default: dist/TorusPlot-<version>.zip)")
    parser.add_argument("--list", action="store_true", help="Only print the files that would be included")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    if args.list:
        print("\n".join(release_files()))
        return 0
    from torusplot import __version__

    out = Path(args.out) if args.out else ROOT / "dist" / f"TorusPlot-{__version__}.zip"
    path = build(out)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(release_files())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
