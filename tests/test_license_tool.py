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
    assert "numpy" in names and "statsmodels" in names
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


def test_lock_file_is_fully_covered_and_in_sync():
    pins = check_licenses.read_lock_names()
    assert len(pins) > 60 and "numpy" in pins and "pywin32" in pins  # all platforms are pinned
    rows, _ = check_licenses.check()
    covered = {r["key"] for r in rows}
    assert set(pins) <= covered, sorted(set(pins) - covered)
    platform = check_licenses.load_section("platform_packages")
    for name, entry in platform.items():
        assert name in pins, f"{name} is listed for another platform but not in the lock file"
        assert str(entry["source_url"]).startswith("https://") and check_licenses.evaluate(entry["verified_license"])[0]
    lock_spec = importlib.util.spec_from_file_location("check_lock", ROOT / "tools" / "check_lock.py")
    check_lock = importlib.util.module_from_spec(lock_spec)
    lock_spec.loader.exec_module(check_lock)
    assert check_lock.main() == 0


def test_test_only_libraries_are_not_runtime_dependencies():
    runtime = set(check_licenses.read_requirement_names(ROOT / "requirements.txt"))
    dev = set(check_licenses.read_requirement_names(ROOT / "requirements-dev.txt"))
    assert {"lifelines", "scikit-posthocs"} <= dev and not {"lifelines", "scikit-posthocs"} & runtime
    assert not {"lifelines", "scikit-posthocs"} & set(check_licenses.read_lock_names())


def test_wheels_with_bundled_binaries_or_copyleft_mentions_have_a_recorded_review():
    reviews = check_licenses.load_section("bundled_review")
    for name in ("numpy", "scipy"):
        found = check_licenses.bundled_findings(name)
        assert found["libs"] or found["mentions"], name  # these wheels do bundle GCC runtime libraries
        assert name in reviews and "libquadmath" in reviews[name]["finding"]
    rows, _ = check_licenses.check()
    for row in rows:
        found = row.get("bundled") or {"libs": [], "mentions": []}
        if found["libs"] or found["mentions"]:
            assert row["key"] in reviews, f"{row['key']} needs a bundled_review entry: {found}"
    for name, entry in reviews.items():
        assert entry.get("finding") and str(entry.get("source_url", "")).startswith("https://"), name


def test_psf_history_wording_is_not_mistaken_for_a_gpl_grant():
    text = "most Python releases have also been GPL-compatible; GPL- from compatible? (1)"
    assert not check_licenses._COPYLEFT_MENTION.search(check_licenses._BENIGN.sub("", text))
    assert check_licenses._COPYLEFT_MENTION.search("License: LGPL-2.1-or-later")


def test_optional_ai_requirements_are_checked_when_installed():
    rows, _ = check_licenses.check()
    optional = [r for r in rows if r.get("optional")]
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return
    assert any(r["key"] == "anthropic" for r in optional)
    assert all(r["ok"] for r in optional)
