#!/usr/bin/env python
"""License gate for TorusPlot.

Fails (exit code 1) if any *runtime* dependency -- every package named in
``requirements.txt`` plus everything those packages pull in -- has a license
outside the permissive allowlist in the project's LICENSE POLICY.

How it works
------------
1. Read the package names in ``requirements.txt``.
2. Walk the installed dependency graph (``importlib.metadata``) to get the full
   runtime closure for the current platform.
3. Ask ``pip-licenses`` (MIT) for each package's license metadata.
4. Evaluate the license expression:
   * ``A OR B``  -> fine if at least one side is allowed
   * ``A AND B`` -> every part must be allowed
   * anything matching a copyleft / non-commercial pattern is rejected
   * unknown or blank metadata is rejected until a human-verified entry exists
     in ``tools/license_overrides.yaml`` (with a source URL).

Usage::

    python tools/check_licenses.py            # check, print a table
    python tools/check_licenses.py --notices  # also (re)write THIRD_PARTY_NOTICES.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
OVERRIDES = ROOT / "tools" / "license_overrides.yaml"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"

# Canonical permissive licenses allowed by the LICENSE POLICY.
ALLOWED = {
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSD",
    "Apache-2.0",
    "ISC",
    "PSF-2.0",
    "HPND",
    "Matplotlib",
    "Unlicense",
    "CC0-1.0",
    "MPL-2.0",
}

# Free-text / classifier spellings seen in package metadata -> canonical name.
ALIASES = {
    "mit": "MIT",
    "mit license": "MIT",
    "expat": "MIT",
    "bsd": "BSD",
    "bsd license": "BSD",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd 2-clause license": "BSD-2-Clause",
    "simplified bsd": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "new bsd": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "modified bsd license": "BSD-3-Clause",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "apache software license 2.0": "Apache-2.0",
    "isc": "ISC",
    "isc license": "ISC",
    "isc license (iscl)": "ISC",
    "psf-2.0": "PSF-2.0",
    "psf": "PSF-2.0",
    "python-2.0": "PSF-2.0",
    "python software foundation license": "PSF-2.0",
    "hpnd": "HPND",
    "historical permission notice and disclaimer (hpnd)": "HPND",
    "matplotlib license": "Matplotlib",
    "unlicense": "Unlicense",
    "the unlicense (unlicense)": "Unlicense",
    "cc0-1.0": "CC0-1.0",
    "cc0 1.0 universal (cc0 1.0) public domain dedication": "CC0-1.0",
    "mpl-2.0": "MPL-2.0",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
}

# Never acceptable, whatever else the metadata says.
DISALLOWED = re.compile(
    r"(\bA?L?GPL|GNU (LESSER |AFFERO )?GENERAL PUBLIC|\bSSPL|SERVER SIDE PUBLIC|\bBUSL|"
    r"BUSINESS SOURCE|COMMONS CLAUSE|NON-?COMMERCIAL|\bCC-BY-NC|-NC\b)",
    re.IGNORECASE,
)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirement_names(path: Path) -> list[str]:
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            names.append(_norm(m.group(1)))
    return names


def runtime_closure(roots: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return ({normalized name: installed version}, [missing packages])."""
    from packaging.requirements import Requirement

    seen: dict[str, str] = {}
    missing: list[str] = []
    stack: list[tuple[str, frozenset[str]]] = [(r, frozenset()) for r in roots]
    visited: set[tuple[str, frozenset[str]]] = set()
    while stack:
        name, extras = stack.pop()
        if (name, extras) in visited:
            continue
        visited.add((name, extras))
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if name not in missing:
                missing.append(name)
            continue
        seen[_norm(dist.metadata["Name"])] = dist.version
        for req_str in dist.requires or []:
            req = Requirement(req_str)
            if req.marker is not None:
                envs = [{"extra": e} for e in extras] or [{"extra": ""}]
                if not any(req.marker.evaluate(env) for env in envs):
                    continue
            stack.append((_norm(req.name), frozenset(req.extras)))
    return seen, missing


def pip_licenses(packages: list[str]) -> dict[str, dict]:
    cmd = [
        sys.executable,
        "-m",
        "piplicenses",
        "--from=mixed",
        "--format=json",
        "--with-urls",
        "--packages",
        *packages,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return {_norm(row["Name"]): row for row in json.loads(out)}


def load_overrides() -> dict[str, dict]:
    import yaml

    if not OVERRIDES.exists():
        return {}
    data = yaml.safe_load(OVERRIDES.read_text(encoding="utf-8")) or {}
    return {_norm(k): v for k, v in (data.get("packages") or {}).items()}


def evaluate(license_text: str) -> tuple[bool, str]:
    """Return (ok, reason) for a license string / SPDX-like expression."""
    text = (license_text or "").strip()
    if not text or text.upper() in {"UNKNOWN", "NONE", "N/A"}:
        return False, "unknown or blank license metadata"
    alternatives = re.split(r"\s+OR\s+", text)
    reasons = []
    for alt in alternatives:
        parts = [p.strip() for p in re.split(r"\s+AND\s+|;", alt) if p.strip(" ()")]
        bad = []
        for part in parts:
            if DISALLOWED.search(part):
                bad.append(f"disallowed: {part}")
                continue
            canon = ALIASES.get(part.lower()) or ALIASES.get(part.strip(" ()").lower())
            canon = canon or part.strip(" ()")
            if canon not in ALLOWED:
                bad.append(f"not on allowlist: {part}")
        if parts and not bad:
            return True, "allowed"
        reasons.extend(bad)
    return False, "; ".join(dict.fromkeys(reasons)) or "could not parse license"


def check() -> tuple[list[dict], list[str]]:
    roots = read_requirement_names(REQUIREMENTS)
    closure, missing = runtime_closure(roots)
    info = pip_licenses(sorted(closure))
    overrides = load_overrides()
    rows = []
    for name in sorted(closure):
        meta = info.get(name, {})
        reported = (meta.get("License") or "").strip()
        ok, reason = evaluate(reported)
        effective = reported
        note = ""
        ov = overrides.get(name)
        if not ok and ov:
            if str(ov.get("license_reported", "")).strip() != reported:
                reason += " (override exists but the reported license changed; re-verify)"
            elif not ov.get("source_url"):
                reason += " (override has no source_url)"
            else:
                ok2, reason2 = evaluate(str(ov.get("verified_license", "")))
                if ok2:
                    ok, reason = True, "allowed via verified override"
                    effective = f"{ov['verified_license']} (reported: {reported or 'blank'})"
                    note = str(ov.get("source_url"))
                else:
                    reason = f"override license rejected: {reason2}"
        rows.append(
            {
                "name": meta.get("Name", name),
                "version": closure[name],
                "license": effective,
                "url": meta.get("URL", "") if meta.get("URL") != "UNKNOWN" else "",
                "ok": ok,
                "reason": reason,
                "direct": name in roots,
                "override_source": note,
            }
        )
    return rows, missing


def write_notices(rows: list[dict]) -> None:
    lines = [
        "# Third-Party Notices",
        "",
        "TorusPlot (Copyright (c) 2026 Torus Washington LLC) uses the open-source packages",
        "below at runtime. They are installed from PyPI by the launcher; TorusPlot does not",
        "modify them. Each package remains under its own license. This file is generated by",
        "`python tools/check_licenses.py --notices`.",
        "",
        "| Package | Version | License | URL | Direct dependency |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        url = r["url"] or f"https://pypi.org/project/{r['name']}/"
        lines.append(
            f"| {r['name']} | {r['version']} | {r['license']} | {url} | "
            f"{'yes' if r['direct'] else 'no (transitive)'} |"
        )
    lines += [
        "",
        "## Notes on compiled wheels",
        "",
        "The binary wheels of NumPy and SciPy published on PyPI include, as separate shared",
        "libraries, the GCC runtime libraries `libgfortran` (GPL-3.0-or-later WITH",
        "GCC-exception-3.1; the exception expressly permits use by proprietary programs) and",
        "`libquadmath` (LGPL-2.1-or-later, dynamically loaded and replaceable). TorusPlot does",
        "not ship these files: they are downloaded from PyPI onto the user's computer by the",
        "launcher. See DECISIONS.md (entry D-007) for what this means if TorusPlot is ever",
        "distributed as a single frozen installer.",
        "",
        "## Bundled fonts",
        "",
        "See `torusplot/plot/fonts/README.md` for the bundled fonts and their licenses",
        "(SIL Open Font License 1.1 / Apache-2.0).",
        "",
    ]
    NOTICES.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--notices", action="store_true", help="write THIRD_PARTY_NOTICES.md")
    parser.add_argument("--quiet", action="store_true", help="only print problems")
    args = parser.parse_args(argv)

    rows, missing = check()
    failures = [r for r in rows if not r["ok"]]
    if not args.quiet:
        width = max(len(r["name"]) for r in rows) + 2
        for r in rows:
            flag = "OK  " if r["ok"] else "FAIL"
            print(f"{flag} {r['name']:<{width}} {r['version']:<14} {r['license']}")
        print(f"\n{len(rows)} runtime packages checked.")
    if missing:
        print("\nNot installed (cannot check): " + ", ".join(sorted(missing)))
    if failures:
        print("\nLICENSE CHECK FAILED:")
        for r in failures:
            print(f"  - {r['name']} {r['version']}: {r['license'] or '(blank)'} -> {r['reason']}")
        print(
            "\nFix: remove the package, or verify its license by hand from its repository and "
            "record it in tools/license_overrides.yaml with a source URL."
        )
    direct_missing = [m for m in missing if m in read_requirement_names(REQUIREMENTS)]
    if args.notices and not failures:
        write_notices(rows)
        print(f"Wrote {NOTICES.name}")
    if failures or direct_missing:
        return 1
    if not args.quiet:
        print("License check passed: all runtime dependencies are on the permissive allowlist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
