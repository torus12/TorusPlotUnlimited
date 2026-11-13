"""The license gate itself must behave as the LICENSE POLICY says."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_licenses", ROOT / "tools" / "check_licenses.py")
check_licenses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_licenses)


@pytest.mark.parametrize(
    "text",
    [
        "MIT",
        "MIT License",
        "BSD License",
        "BSD-3-Clause",
        "Apache-2.0",
        "Apache Software License",
        "Apache-2.0 OR BSD-3-Clause",
        "Python Software Foundation License",
        "Mozilla Public License 2.0 (MPL 2.0)",
        "MIT AND BSD-3-Clause",
        "GPL-3.0-only OR MIT",
    ],
)
def test_allowed_licenses_pass(text):
    ok, reason = check_licenses.evaluate(text)
    assert ok, reason


@pytest.mark.parametrize(
    "text",
    [
        "",
        "UNKNOWN",
        "GPL-3.0",
        "GNU General Public License v3 (GPLv3)",
        "LGPL-2.1-or-later",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "AGPL-3.0",
        "SSPL-1.0",
        "BUSL-1.1",
        "MIT AND GPL-2.0",
        "CC-BY-NC-4.0",
        "MIT with Commons Clause",
        "Some Custom License",
    ],
)
def test_disallowed_or_unknown_licenses_fail(text):
    ok, _ = check_licenses.evaluate(text)
    assert not ok


def test_requirements_parse():
    names = check_licenses.read_requirement_names(ROOT / "requirements.txt")
    assert "numpy" in names and "scikit-posthocs" in names
    assert "pingouin" not in names and "figrecipe" not in names


def test_runtime_closure_passes_policy():
    rows, missing = check_licenses.check()
    direct = set(check_licenses.read_requirement_names(ROOT / "requirements.txt"))
    assert not [m for m in missing if m in direct], f"direct deps not installed: {missing}"
    failures = [(r["name"], r["license"], r["reason"]) for r in rows if not r["ok"]]
    assert not failures, failures
    assert len(rows) > 30


def test_overrides_have_source_urls():
    for name, entry in check_licenses.load_overrides().items():
        assert str(entry.get("source_url", "")).startswith("https://"), name
        assert entry.get("verified_license"), name
        assert entry.get("license_reported") is not None, name
