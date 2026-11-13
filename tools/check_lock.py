#!/usr/bin/env python
"""Verify that requirements-lock.txt still satisfies requirements.txt.

    python tools/check_lock.py

The lock file pins the exact, hash-checked versions the launcher installs (the set that is
tested in CI and covered by the license audit). Regenerate it after changing
requirements.txt:

    uv pip compile requirements.txt --universal --python-version 3.11 --generate-hashes \\
        --no-header -o requirements-lock.txt

then run `python tools/check_licenses.py --notices`.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    from packaging.requirements import Requirement
    from packaging.version import Version

    lock_text = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    pins: dict[str, list[Version]] = {}
    for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", lock_text, re.MULTILINE):
        pins.setdefault(_norm(m.group(1)), []).append(Version(m.group(2)))
    problems = []
    if "--hash=sha256:" not in lock_text:
        problems.append("requirements-lock.txt has no hashes")
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        versions = pins.get(_norm(req.name))
        if not versions:
            problems.append(f"{req.name} is in requirements.txt but not pinned in requirements-lock.txt")
            continue
        for v in versions:
            if not req.specifier.contains(v, prereleases=True):
                problems.append(f"lock pins {req.name}=={v}, which requirements.txt ({req.specifier}) does not allow")
    if problems:
        print("LOCK FILE OUT OF SYNC:\n  - " + "\n  - ".join(problems))
        print(__doc__)
        return 1
    print(f"Lock file OK: {len(pins)} pinned packages, hashes present, consistent with requirements.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
