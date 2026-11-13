"""Documentation and repository hygiene: docs exist, stay in sync with the code, and respect the
naming / licensing policy."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

REQUIRED_FILES = [
    "LICENSE", "README.md", "DECISIONS.md", "PROGRESS.md", "THIRD_PARTY_NOTICES.md", ".env.example",
    "requirements.txt", "requirements-dev.txt", "run_torusplot.bat", "run_torusplot.sh", "cli.py",
    "docs/USER_GUIDE.md", "docs/STATS_METHODS.md", "docs/MCP.md", "docs/PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md",
    "docs/REFERENCE_NOTES.md", "docs/MODEL_LICENSE_RESEARCH.md", "examples/README.md",
    "tests/fixtures/SOURCES.md", "torusplot/plot/fonts/README.md", ".github/workflows/ci.yml",
]

#: Trademarks that must not appear in the product (UI, code, presets, examples, user docs).
BANNED = re.compile(r"graph\s?pad|\bprism\b", re.IGNORECASE)
#: Files that may name other products because they are provenance / policy records.
ALLOWED_TO_NAME = {"docs/REFERENCE_NOTES.md", "DECISIONS.md", "FINAL_REPORT.md", "tests/test_docs.py",
                   "tests/stats/test_registry_integration.py", "tests/plot/test_styles_palettes.py"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".bat", ".sh", ".svg", ".json", ".js", ".csv", ".cfg"}


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / line for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:  # not a git checkout (e.g. downloaded ZIP)
        skip = {".venv", "_reference", ".git", "__pycache__", "build", "dist", "output"}
        files = [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & skip)]
    return [f for f in files if f.exists()]


@pytest.mark.parametrize("rel", REQUIRED_FILES)
def test_required_files_exist(rel):
    path = ROOT / rel
    assert path.exists() and path.stat().st_size > 0, rel


def test_license_is_proprietary():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("Copyright (c) 2026 Torus Washington LLC. All rights reserved.")
    assert "MIT License" not in text and "GNU" not in text


def test_no_trademarked_product_names_in_shipped_files():
    offenders = []
    for path in tracked_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED_TO_NAME or path.suffix.lower() not in TEXT_SUFFIXES or rel.startswith("tests/fixtures/nist/"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for n, line in enumerate(text.splitlines(), start=1):
            if BANNED.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()[:80]}")
    assert not offenders, "\n".join(offenders[:20])


def test_reference_material_and_secrets_are_not_tracked():
    rels = [p.relative_to(ROOT).as_posix() for p in tracked_files()]
    assert not [r for r in rels if r.startswith("_reference/")]
    assert ".env" not in rels
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "_reference/" in ignore and ".env" in ignore
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if line.strip().endswith("_API_KEY="):
            continue
        assert not re.match(r"^[A-Z_]*API_KEY=\S+", line), "no real keys in .env.example"


def test_no_imports_of_disallowed_packages():
    banned = re.compile(r"^\s*(import|from)\s+(pingouin|figrecipe)\b", re.MULTILINE)
    for path in tracked_files():
        if path.suffix == ".py":
            assert not banned.search(path.read_text(encoding="utf-8", errors="ignore")), path
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "pingouin" not in reqs and "figrecipe" not in reqs


def test_every_analysis_is_documented():
    from torusplot.stats import load_all

    guide = (DOCS / "USER_GUIDE.md").read_text(encoding="utf-8")
    methods = (DOCS / "STATS_METHODS.md").read_text(encoding="utf-8")
    for key, info in load_all().items():
        assert info.label in guide, f"USER_GUIDE.md has no worked example for '{info.label}' ({key})"
        assert key in methods, f"STATS_METHODS.md does not document '{key}'"


def test_user_guide_examples_are_generated_from_real_output():
    script = ROOT / "tools" / "build_user_guide.py"
    result = subprocess.run([sys.executable, str(script), "--check"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_readme_links_and_images_resolve():
    for doc in [ROOT / "README.md", DOCS / "USER_GUIDE.md", DOCS / "MCP.md"]:
        text = doc.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text) + re.findall(r'src="([^"]+)"', text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (doc.parent / target).exists(), f"{doc.name}: broken link or image '{target}'"


def test_windows_launcher_is_safe_for_cmd():
    raw = (ROOT / "run_torusplot.bat").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), "batch file must use CRLF line endings"
    text = raw.decode("ascii")  # plain ASCII only: cmd.exe code pages vary
    assert "requirements.txt" in text and "streamlit run app\\streamlit_app.py" in text and "-m venv .venv" in text
    depth = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("rem") or not stripped:
            continue
        outside_quotes = re.sub(r'"[^"]*"', "", stripped)
        if depth > 0 and stripped.lower().startswith("echo"):
            assert "(" not in outside_quotes and ")" not in outside_quotes, f"unescaped parenthesis inside a block: {line}"
        depth += outside_quotes.count("(") - outside_quotes.count(")")
    assert depth == 0, "unbalanced parentheses in run_torusplot.bat"


def test_third_party_notices_cover_all_requirements():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8").lower()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        name = re.split(r"[<>=!~\s#]", line.strip(), maxsplit=1)[0].lower()
        if name:
            assert f"| {name} |" in notices or f"| {name.replace('-', '_')} |" in notices, name
    for bad in ("gpl", "agpl", "sspl", "busl", "non-commercial"):
        table = "\n".join(line for line in notices.splitlines() if line.startswith("|"))
        assert not re.search(rf"\|\s[^|]*\b{bad}\b", table), f"'{bad}' appears in the dependency table"
