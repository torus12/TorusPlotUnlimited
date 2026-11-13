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

It also checks three things that package metadata alone cannot show:

* **The lock file.** Every package pinned in ``requirements-lock.txt`` (all platforms) must
  be covered: installed here and passing, or listed under ``platform_packages`` in the
  overrides file with a verified license (e.g. Windows-only ``pywin32``).
* **Binaries and notices inside wheels.** If an installed runtime package ships a ``*.libs``
  folder of bundled shared libraries, or its license texts mention a GPL-family license, a
  human review must be recorded under ``bundled_review`` in the overrides file.
* **Optional extras.** ``requirements-ai.txt`` and ``requirements-mcp.txt`` are checked
  whenever their packages are installed, and their lock file is covered like the main one.

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
#: Optional extras (not installed by the launcher): cloud-AI SDK and the MCP server SDK.
OPTIONAL_REQUIREMENTS = [ROOT / "requirements-ai.txt", ROOT / "requirements-mcp.txt"]
LOCK = ROOT / "requirements-lock.txt"
OPTIONAL_LOCK = ROOT / "requirements-optional-lock.txt"
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


def _load_override_file() -> dict:
    import yaml

    if not OVERRIDES.exists():
        return {}
    return yaml.safe_load(OVERRIDES.read_text(encoding="utf-8")) or {}


def load_overrides() -> dict[str, dict]:
    return {_norm(k): v for k, v in (_load_override_file().get("packages") or {}).items()}


def load_section(section: str) -> dict[str, dict]:
    """A named section of the overrides file (``platform_packages`` / ``bundled_review``)."""
    return {_norm(k): v for k, v in (_load_override_file().get(section) or {}).items()}


def read_lock_names(path: Path = LOCK) -> dict[str, list[str]]:
    """{normalized name: [pinned versions]} from the hashed lock file (all platforms)."""
    pins: dict[str, list[str]] = {}
    if not path.exists():
        return pins
    for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", path.read_text(encoding="utf-8"), re.MULTILINE):
        pins.setdefault(_norm(m.group(1)), []).append(m.group(2))
    return pins


_COPYLEFT_MENTION = re.compile(r"\b(?:A|L)?GPL\b|GNU (?:LESSER |AFFERO )?GENERAL PUBLIC", re.IGNORECASE)
#: The PSF license text describes Python's history with the phrase "GPL-compatible"; that is not
#: a GPL grant, so it is removed before scanning.
_BENIGN = re.compile(r"GPL-\s*(?:from\s+)?compatible(?:\?)?", re.IGNORECASE)


def bundled_findings(name: str) -> dict[str, list[str]]:
    """What an installed wheel carries beyond its own code: bundled shared-library folders
    and GPL-family mentions in its license texts."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"libs": [], "mentions": []}
    files = dist.files or []
    libs = sorted({str(f).replace("\\", "/").split("/")[0] for f in files
                   if str(f).replace("\\", "/").split("/")[0].endswith((".libs", ".dylibs"))})
    texts = [dist.read_text("METADATA") or ""]
    for f in files:
        path = str(f).replace("\\", "/")
        if ".dist-info/" in path and re.search(r"licen[sc]e|copying|notice", path, re.IGNORECASE):
            try:
                texts.append(f.read_text() or "")
            except (OSError, UnicodeDecodeError):
                continue
    mentions = sorted({m.group(0).upper() for t in texts for m in _COPYLEFT_MENTION.finditer(_BENIGN.sub("", t))})
    return {"libs": libs, "mentions": mentions}


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


def _rows_for(roots: list[str], overrides: dict[str, dict], optional: bool = False) -> tuple[list[dict], list[str]]:
    closure, missing = runtime_closure(roots)
    info = pip_licenses(sorted(closure)) if closure else {}
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
                "key": name,
                "version": closure[name],
                "license": effective,
                "url": meta.get("URL", "") if meta.get("URL") != "UNKNOWN" else "",
                "ok": ok,
                "reason": reason,
                "direct": name in roots,
                "optional": optional,
                "override_source": note,
            }
        )
    return rows, missing


def check() -> tuple[list[dict], list[str]]:
    """Check the runtime closure (and the optional extras when installed).

    Returns (rows, missing). A row with ``ok == False`` fails the gate. Rows cover: installed
    runtime packages, lock-file packages for other platforms, bundled-binary reviews.
    """
    overrides = load_overrides()
    roots = read_requirement_names(REQUIREMENTS)
    rows, missing = _rows_for(roots, overrides)
    seen = {r["key"] for r in rows}

    # optional extras (Anthropic SDK, MCP SDK): checked whenever they are installed
    for req_file in OPTIONAL_REQUIREMENTS:
        if not req_file.exists():
            continue
        opt_rows, _ = _rows_for(read_requirement_names(req_file), overrides, optional=True)
        for r in opt_rows:
            if r["key"] not in seen:
                rows.append(r)
                seen.add(r["key"])

    # lock files: every pinned package (any platform) must be covered
    platform = load_section("platform_packages")
    core_pins = read_lock_names(LOCK)
    all_pins = {**read_lock_names(OPTIONAL_LOCK), **core_pins}
    for name, versions in sorted(all_pins.items()):
        if name in seen:
            continue
        entry = platform.get(name)
        if entry and entry.get("source_url"):
            ok, reason = evaluate(str(entry.get("verified_license", "")))
            reason = "allowed via platform_packages (not installed on this platform)" if ok else f"platform_packages license rejected: {reason}"
        else:
            ok, reason = False, "pinned in requirements-lock.txt but not installed here and not listed under platform_packages"
        rows.append({"name": name, "key": name, "version": ", ".join(versions), "ok": ok, "reason": reason,
                     "license": f"{(entry or {}).get('verified_license', '?')} ({(entry or {}).get('when', 'other platforms')})",
                     "url": (entry or {}).get("source_url", ""), "direct": False, "optional": name not in core_pins,
                     "override_source": (entry or {}).get("source_url", "")})
        seen.add(name)

    # binaries / copyleft mentions inside installed wheels need a recorded human review
    reviews = load_section("bundled_review")
    for row in rows:
        found = bundled_findings(row["key"])
        row["bundled"] = found
        if (found["libs"] or found["mentions"]) and row["ok"]:
            review = reviews.get(row["key"])
            if not review or not review.get("finding") or not review.get("source_url"):
                row["ok"] = False
                row["reason"] = (
                    f"wheel needs a bundled-component review (libs: {found['libs'] or 'none'}; license text "
                    f"mentions: {found['mentions'] or 'none'}): add it under bundled_review in tools/license_overrides.yaml"
                )
    return rows, missing


def write_notices(rows: list[dict]) -> None:
    pins = {**read_lock_names(OPTIONAL_LOCK), **read_lock_names(LOCK)}
    lines = [
        "# Third-Party Notices",
        "",
        "TorusPlot (Copyright (c) 2026 Torus Washington LLC) uses the open-source packages",
        "below at runtime. They are installed from PyPI by the launcher; TorusPlot does not",
        "modify or redistribute them. Each package remains under its own license, and each",
        "installed package carries its own license text in its `*.dist-info` folder.",
        "",
        "This file is generated by `python tools/check_licenses.py --notices`. Versions are the",
        "ones pinned in `requirements-lock.txt` (and `requirements-optional-lock.txt` for the",
        "optional extras); where two versions are shown, the lock file selects one by Python",
        "version. Licenses are read from each",
        "package's metadata, with hand-verified entries from `tools/license_overrides.yaml`.",
        "",
        "## Runtime dependencies",
        "",
        "| Package | Version(s) | License | URL | Role |",
        "|---|---|---|---|---|",
    ]

    def role(r: dict) -> str:
        if r.get("optional"):
            return "optional extra (requirements-ai.txt / requirements-mcp.txt; not installed by default)"
        return "direct dependency" if r["direct"] else "transitive"

    for r in rows:
        url = r["url"] or f"https://pypi.org/project/{r['name']}/"
        version = ", ".join(pins.get(r["key"], [])) or r["version"]
        lines.append(f"| {r['name']} | {version} | {r['license']} | {url} | {role(r)} |")
    reviews = load_section("bundled_review")
    lines += [
        "",
        "## Components bundled inside binary wheels",
        "",
        "Some PyPI wheels contain third-party native libraries or carry several license texts.",
        "Each such wheel was reviewed by hand (recorded in `tools/license_overrides.yaml`,",
        "section `bundled_review`); the license gate fails if a new one appears unreviewed.",
        "Wheels differ between Windows, macOS and Linux, so the gate runs on each CI platform.",
        "",
        "| Package | Finding |",
        "|---|---|",
    ]
    for name in sorted(reviews):
        finding = " ".join(str(reviews[name].get("finding", "")).split())
        lines.append(f"| {name} | {finding} |")
    lines += [
        "",
        "TorusPlot does not ship any of these files: they are downloaded from PyPI onto the",
        "user's computer by the launcher. See DECISIONS.md (D-007) and",
        "docs/COMMERCIAL_RELEASE_CHECKLIST.md for what changes if TorusPlot is ever distributed",
        "as a single frozen installer that bundles the wheels.",
        "",
        "## Bundled fonts",
        "",
        "Inter, Source Sans 3 and Liberation Sans are bundled unmodified under the SIL Open Font",
        "License 1.1 (OFL-1.1). The license texts ship next to the font files; see",
        "`torusplot/plot/fonts/README.md`.",
        "",
        "## Test-only and developer tools (not shipped, not installed by the launcher)",
        "",
        "`requirements-dev.txt` lists tools used only to develop and validate TorusPlot: pytest,",
        "pytest-cov, ruff, pip-licenses, hypothesis, and two independent statistics libraries",
        "(lifelines, scikit-posthocs) that the test-suite uses to cross-check TorusPlot's own",
        "implementations.",
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
            tag = " [optional]" if r.get("optional") else ""
            print(f"{flag} {r['name']:<{width}} {r['version']:<14} {r['license']}{tag}")
        print(f"\n{len(rows)} packages checked (runtime closure, lock file, optional extras, bundled binaries).")
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
