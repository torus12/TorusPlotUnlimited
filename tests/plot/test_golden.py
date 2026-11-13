"""Golden-image tests: every plot type and every style preset must keep its look.

Images are compared with a tolerance (root-mean-square pixel difference) so tiny
anti-aliasing differences between platforms do not fail the suite. To deliberately
change the expected look run ``python tools/regenerate_goldens.py --reason "..."``.
"""

import os
from pathlib import Path

import pytest
from matplotlib.testing.compare import compare_images

from tests.golden.cases import DPI, build_cases
from torusplot.export import figure_bytes
from torusplot.plot import figure_warnings, preset_names
from torusplot.spec import PLOT_TYPES

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "images"
FAILED = GOLDEN.parent / "_failed"
TOL = float(os.environ.get("TORUSPLOT_GOLDEN_TOL", "12"))
CASES = build_cases()


def test_every_plot_type_and_preset_has_a_golden_case():
    for plot_type in PLOT_TYPES:
        assert plot_type in CASES, f"no golden case for plot type {plot_type}"
    assert sum(name.startswith("style_") for name in CASES) == len(preset_names())
    missing = [name for name in CASES if not (GOLDEN / f"{name}.png").exists()]
    assert not missing, f"golden images missing (run tools/regenerate_goldens.py): {missing}"


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_image(name, tmp_path):
    fig = CASES[name]()
    assert figure_warnings(fig) == [], figure_warnings(fig)
    actual = tmp_path / f"{name}.png"
    actual.write_bytes(figure_bytes(fig, "png", dpi=DPI))
    expected = GOLDEN / f"{name}.png"
    diff = compare_images(str(expected), str(actual), tol=TOL)
    if diff is not None:
        FAILED.mkdir(exist_ok=True)
        (FAILED / f"{name}-actual.png").write_bytes(actual.read_bytes())
        pytest.fail(f"{name} differs from its golden image (tolerance {TOL}):\n{diff}")
