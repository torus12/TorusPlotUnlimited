#!/usr/bin/env python
"""Verify that requirements-lock.txt still satisfies requirements.txt.

    python tools/check_lock.py

The lock file pins the exact, hash-checked versions the launcher installs (the set that is
tested in CI and covered by the license audit). Regenerate it after changing
requirements.txt:

    uv pip compile requirements.txt --universal --python-version 3.11 --generate-hashes \\
        --no-header -o requirements-lock.txt

    uv pip compile requirements.txt requirements-ai.txt requirements-mcp.txt \\
        -c requirements-lock.txt --universal --python-version 3.11 --generate-hashes \\
        --no-header -o requirements-optional-lock.txt

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

    def read_pins(lock: str) -> dict[str, list[Version]]:
        text = (ROOT / lock).read_text(encoding="utf-8")
        if "--hash=sha256:" not in text:
            problems.append(f"{lock} has no hashes")
        found: dict[str, list[Version]] = {}
        for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", text, re.MULTILINE):
            found.setdefault(_norm(m.group(1)), []).append(Version(m.group(2)))
        return found

    def check_file(req_file: str, lock: str, pinned: dict[str, list[Version]]) -> None:
        for raw in (ROOT / req_file).read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            req = Requirement(line)
            versions = pinned.get(_norm(req.name))
            if not versions:
                problems.append(f"{req.name} is in {req_file} but not pinned in {lock}")
                continue
            for v in versions:
                if not req.specifier.contains(v, prereleases=True):
                    problems.append(f"{lock} pins {req.name}=={v}, which {req_file} ({req.specifier}) does not allow")

    problems: list[str] = []
    pins = read_pins("requirements-lock.txt")
    optional = read_pins("requirements-optional-lock.txt")
    check_file("requirements.txt", "requirements-lock.txt", pins)
    for req_file in ("requirements.txt", "requirements-ai.txt", "requirements-mcp.txt"):
        check_file(req_file, "requirements-optional-lock.txt", optional)
    for name, versions in pins.items():  # the optional lock must not move any core version
        if sorted(optional.get(name, [])) != sorted(versions):
            problems.append(f"{name}: core lock pins {versions} but the optional lock pins {optional.get(name)}")
    if problems:
        print("LOCK FILE OUT OF SYNC:\n  - " + "\n  - ".join(problems))
        print(__doc__)
        return 1
    print(f"Lock files OK: {len(pins)} core + {len(optional) - len(pins)} optional pinned packages, hashes present, "
          "consistent with the requirements files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
