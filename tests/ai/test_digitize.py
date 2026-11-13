"""Digitizer tests on synthetic images with known truth.

Accuracy claim checked here: recovered values are within 1.5 % of the axis range (of the
log10 range on logarithmic axes) for clean, non-overlapping markers, curves and bars.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tests.ai import PlotImage, flat_figure, png_of
from torusplot.ai import digitize as dg
from torusplot.ai.digitize import (
    APPROXIMATE_NOTE,
    AxisCalibration,
    DigitizeError,
    DigitizeResult,
    calibration_from_plot_area,
    candidate_colors,
    detect_plot_area,
    digitize,
    pixel_to_data,
    to_table,
)
from torusplot.data import XYTable

RED, BLUE, GREEN = "#D62728", "#1F77B4", "#2CA02C"


# ====================================================================== calibration maths
def test_linear_calibration():
    axis = AxisCalibration(100, 0.0, 500, 10.0)
    assert axis.to_data(100) == 0.0 and axis.to_data(500) == 10.0 and axis.to_data(300) == 5.0
    assert axis.to_data(0) == -2.5  # extrapolates
    assert axis.to_pixel(7.5) == 400.0
    np.testing.assert_allclose(axis.to_data(np.array([100, 200, 500])), [0.0, 2.5, 10.0])
    np.testing.assert_allclose(axis.to_pixel(axis.to_data(np.array([123.4, 456.7]))), [123.4, 456.7])
    # an inverted axis (image rows grow downwards)
    y = AxisCalibration(400, 0.0, 50, 70.0)
    assert y.to_data(400) == 0.0 and y.to_data(50) == 70.0 and y.to_data(225) == 35.0


def test_log_calibration_interpolates_in_log_space():
    axis = AxisCalibration(100, 1.0, 400, 1000.0, log=True)
    assert axis.to_data(100) == pytest.approx(1.0) and axis.to_data(400) == pytest.approx(1000.0)
    assert axis.to_data(200) == pytest.approx(10.0) and axis.to_data(300) == pytest.approx(100.0)
    assert axis.to_data(250) == pytest.approx(10 ** 1.5)
    assert axis.to_pixel(100.0) == pytest.approx(300.0)
    np.testing.assert_allclose(axis.to_pixel(np.array([1.0, 10.0])), [100.0, 200.0])


def test_pixel_to_data_and_plot_area_calibration():
    x_axis, y_axis = calibration_from_plot_area((50, 20, 450, 320), 0, 100, 1, 1000, y_log=True)
    assert pixel_to_data(50, 320, x_axis, y_axis) == (0.0, pytest.approx(1.0))
    assert pixel_to_data(450, 20, x_axis, y_axis) == (100.0, pytest.approx(1000.0))
    x, y = pixel_to_data(250, 220, x_axis, y_axis)
    assert x == 50.0 and y == pytest.approx(10.0)
    xs, ys = pixel_to_data(np.array([50, 450]), np.array([320, 20]), x_axis, y_axis)
    np.testing.assert_allclose(xs, [0, 100])
    np.testing.assert_allclose(ys, [1, 1000])


@pytest.mark.parametrize("args,message", [
    ((10, 0.0, 10, 5.0), "different pixels"),
    ((10, 3.0, 20, 3.0), "different values"),
    ((10, float("nan"), 20, 3.0), "finite numbers"),
    ((10, "a", 20, 3.0), "finite numbers"),
])
def test_invalid_calibrations(args, message):
    with pytest.raises(DigitizeError, match=message):
        AxisCalibration(*args)


def test_log_calibration_needs_positive_values():
    with pytest.raises(DigitizeError, match="greater than zero"):
        AxisCalibration(10, 0.0, 20, 100.0, log=True)
    assert AxisCalibration(np.int64(10), np.float64(1.0), 20, 100.0, log=True).to_data(15) == pytest.approx(10.0)


# ====================================================================== plot area and colours
def test_detect_plot_area_on_a_crisp_image():
    image = flat_figure()
    assert detect_plot_area(image) == (50, 20, 380, 250)
    left, top, right, bottom = detect_plot_area(png_of(image), inner=True)
    assert left > 51 and bottom < 250 and (top, right) == (20, 380)  # open sides are not shrunk
    boxed = detect_plot_area(flat_figure(box=True), inner=True)
    assert boxed[1] > 21 and boxed[2] < 379
    assert detect_plot_area(Image.new("RGB", (300, 200), "#FFFFFF")) is None


def test_detect_plot_area_matches_matplotlib_axes():
    plot = PlotImage()
    plot.ax.plot([1, 5, 9], [10, 50, 90], "o", color=RED)
    left, top, right, bottom = detect_plot_area(plot.png())
    x0, y0 = plot.pixel(0.0, 0.0)
    x1, y1 = plot.pixel(10.0, 100.0)
    assert abs(left - x0) <= 2 and abs(bottom - y0) <= 2 and abs(right - x1) <= 2 and abs(top - y1) <= 2


def test_candidate_colors_lists_series_colours_first():
    plot = PlotImage()
    plot.ax.plot(np.linspace(1, 9, 30), np.linspace(10, 90, 30), "s", color=RED, ms=7)
    plot.ax.plot(np.linspace(1, 9, 200), 50 + 30 * np.sin(np.linspace(1, 9, 200)), "-", color=BLUE, lw=2.5)
    colors = candidate_colors(plot.png(), k=6)
    assert {c["hex"] for c in colors if c["role"] == "data"} == {RED, BLUE}
    assert colors[-1]["role"] == "ink" and colors[-1]["hex"] == "#000000"
    assert all(c["role"] != "background" for c in colors)
    assert len(candidate_colors(plot.png(), k=1)) == 1


# ====================================================================== points
def _points_plot(xlog, ylog, seed=0, n=14):
    rng = np.random.default_rng(seed)
    xlim = (1.0, 1000.0) if xlog else (0.0, 10.0)
    ylim = (0.1, 100.0) if ylog else (0.0, 100.0)
    plot = PlotImage(xlog=xlog, ylog=ylog, xlim=xlim, ylim=ylim)
    xs = np.geomspace(2, 600, n) if xlog else np.linspace(0.6, 9.4, n)
    ys = np.geomspace(0.3, 60, n)[rng.permutation(n)] if ylog else rng.uniform(8, 92, n)
    plot.ax.plot(xs, ys, "o", color=RED, ms=6, linestyle="none")
    # a second series in another colour that must be ignored
    plot.ax.plot(xs, ys[::-1], "^", color=BLUE, ms=6, linestyle="none")
    return plot, xs, ys


@pytest.mark.parametrize("xlog,ylog", [(False, False), (True, False), (False, True), (True, True)])
def test_points_mode_within_tolerance(xlog, ylog):
    plot, xs, ys = _points_plot(xlog, ylog)
    x_axis, y_axis = plot.calibrations()
    result = digitize(plot.png(), x_axis, y_axis, color=RED, mode="points", series_name="Red series")
    assert isinstance(result, DigitizeResult)
    assert result.n == len(xs) == len(result.points) == len(result.pixel_points)
    assert list(result.points.columns) == ["x", "y"] and list(result.pixel_points.columns) == ["px", "py"]
    assert result.points["x"].is_monotonic_increasing
    assert plot.error("x", result.points["x"], xs).max() < plot.tolerance("x")
    assert plot.error("y", result.points["y"], ys).max() < plot.tolerance("y")
    assert result.mode == "points" and result.color == RED and result.series_name == "Red series"
    assert result.approximate is True and result.notes == [APPROXIMATE_NOTE]
    assert result.plot_area is not None
    # the pixel points are where the markers really are
    truth = np.array([plot.pixel(x, y) for x, y in zip(xs, ys)])
    assert np.abs(result.pixel_points.to_numpy() - truth).max() < 1.0


def test_points_mode_is_deterministic_and_ignores_other_colours():
    plot, xs, ys = _points_plot(False, False, seed=3)
    x_axis, y_axis = plot.calibrations()
    png = plot.png()
    first = digitize(png, x_axis, y_axis, color=RED)
    second = digitize(png, x_axis, y_axis, color=RED.lower())
    assert first.points.equals(second.points)
    blue = digitize(png, x_axis, y_axis, color=BLUE)
    assert blue.n == len(xs)
    assert plot.error("y", blue.points["y"], ys[::-1]).max() < plot.tolerance("y")


def _dots(centres, radius=6, size=(400, 300), color=RED):
    image = Image.new("RGB", size, "#FFFFFF")
    draw = ImageDraw.Draw(image)
    for cx, cy in centres:
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    return image


def test_overlapping_markers_are_split_and_noted():
    centres = [(60, 200), (110, 150), (160, 220), (210, 90), (260, 180), (300, 120), (309, 120)]  # last two overlap
    image = _dots(centres)
    image.putpixel((350, 50), (214, 39, 40))  # a single stray pixel in the series colour
    x_axis, y_axis = AxisCalibration(0, 0.0, 400, 400.0), AxisCalibration(300, 0.0, 0, 300.0)
    result = digitize(png_of(image), x_axis, y_axis, color=RED, plot_area=(0, 0, 399, 299))
    assert result.n == 7
    np.testing.assert_allclose(result.pixel_points["px"], [c[0] for c in centres], atol=2.0)
    np.testing.assert_allclose(result.pixel_points["py"], [c[1] for c in centres], atol=2.0)
    notes = " | ".join(result.notes)
    assert "1 blob(s) looked like overlapping markers and were split into 2 points" in notes
    assert "overlaps are a guess" in notes
    assert "1 tiny speck(s) in the series colour were ignored" in notes
    assert result.notes[0] == APPROXIMATE_NOTE


def test_points_mode_notes_for_lines_missing_axes_and_limits():
    line = Image.new("RGB", (400, 300), "#FFFFFF")
    ImageDraw.Draw(line).line([(20, 250), (380, 40)], fill=RED, width=4)
    x_axis, y_axis = AxisCalibration(0, 0.0, 400, 4.0), AxisCalibration(300, 0.0, 0, 3.0)
    result = digitize(png_of(line), x_axis, y_axis, color=RED)
    assert result.n == 1
    notes = " | ".join(result.notes)
    assert "plot area could not be detected" in notes and "use curve mode instead" in notes

    many = _dots([(40 + 30 * i, 150) for i in range(10)])
    limited = digitize(png_of(many), x_axis, y_axis, color=RED, max_points=4)
    assert limited.n == 4 and "only the first 4 (by x) are kept" in " | ".join(limited.notes)
    np.testing.assert_allclose(limited.pixel_points["px"], [40, 70, 100, 130], atol=1.0)


# ====================================================================== curve
@pytest.mark.parametrize("xlog,ylog", [(False, False), (True, True), (True, False)])
def test_curve_mode_within_tolerance(xlog, ylog):
    xlim = (1.0, 1000.0) if xlog else (0.0, 10.0)
    ylim = (0.1, 100.0) if ylog else (0.0, 100.0)
    plot = PlotImage(xlog=xlog, ylog=ylog, xlim=xlim, ylim=ylim)
    xx = np.geomspace(1.5, 800, 500) if xlog else np.linspace(0.5, 9.5, 500)
    if ylog:
        def curve(x):
            return 0.5 * np.asarray(x) ** 0.7
    elif xlog:
        def curve(x):
            return 10 + 80 / (1 + (np.asarray(x) / 40.0) ** 1.2)  # a dose-response shape
    else:
        def curve(x):
            return 50 + 40 * np.sin(np.asarray(x))
    plot.ax.plot(xx, curve(xx), "-", color=BLUE, lw=2)
    plot.ax.plot(xx, np.full_like(xx, ylim[1] * 0.5 if not ylog else 3.0), "--", color=GREEN, lw=2)  # distractor
    x_axis, y_axis = plot.calibrations()
    result = digitize(plot.png(), x_axis, y_axis, color=BLUE, mode="curve", max_points=60)
    assert result.mode == "curve" and 50 <= result.n <= 60
    x = result.points["x"].to_numpy()
    assert x[0] == pytest.approx(xx[0], rel=0.05) and x[-1] == pytest.approx(xx[-1], rel=0.05)
    assert np.all(np.diff(x) > 0)
    spacing = np.diff(result.pixel_points["px"].to_numpy())
    assert spacing.max() - spacing.min() <= 2  # evenly spaced columns
    assert plot.error("y", result.points["y"], curve(x)).max() < plot.tolerance("y")
    assert result.notes == [APPROXIMATE_NOTE]


def test_curve_mode_default_sampling_and_ambiguity_note():
    plot = PlotImage()
    xx = np.linspace(0.5, 9.5, 300)
    plot.ax.plot(xx, 20 + 6 * xx, "-", color=BLUE, lw=2)
    x_axis, y_axis = plot.calibrations()
    result = digitize(plot.png(), x_axis, y_axis, color=BLUE, mode="curve")
    assert result.n == 200  # default number of samples
    assert plot.error("y", result.points["y"], 20 + 6 * result.points["x"]).max() < plot.tolerance("y")
    # two separate lines of the same colour: the median is ambiguous and the result says so
    plot.ax.plot(xx, 95 - 6 * xx + 0 * xx, "-", color=BLUE, lw=2)
    plot.ax.plot(xx, np.full_like(xx, 5.0), "-", color=BLUE, lw=2)
    crossed = digitize(plot.png(), x_axis, y_axis, color=BLUE, mode="curve")
    assert "several separate pieces of the series colour" in " | ".join(crossed.notes)


# ====================================================================== bars
def test_bars_mode_linear_with_a_negative_bar():
    plot = PlotImage(xlim=(0.0, 6.0), ylim=(-40.0, 100.0))
    positions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    heights = np.array([20.0, 75.0, -25.0, 50.0, 90.0])
    plot.ax.bar(positions, heights, width=0.6, color=GREEN)
    plot.ax.errorbar(positions, heights, yerr=6, fmt="none", ecolor="#000000", capsize=3)  # ignored: other colour
    x_axis, y_axis = plot.calibrations()
    result = digitize(plot.png(), x_axis, y_axis, color=GREEN, mode="bars")
    assert result.n == 5 and result.mode == "bars"
    assert plot.error("x", result.points["x"], positions).max() < plot.tolerance("x")
    assert plot.error("y", result.points["y"], heights).max() < plot.tolerance("y")


def test_bars_mode_log_axis():
    plot = PlotImage(ylog=True, xlim=(0.0, 5.0), ylim=(1.0, 1000.0))
    positions = np.array([1.0, 2.0, 3.0, 4.0])
    heights = np.array([3.0, 30.0, 300.0, 80.0])
    plot.ax.bar(positions, heights, width=0.5, color=BLUE)
    x_axis, y_axis = plot.calibrations()
    result = digitize(plot.png(), x_axis, y_axis, color=BLUE, mode="bars")
    assert result.n == 4
    assert plot.error("x", result.points["x"], positions).max() < plot.tolerance("x")
    assert plot.error("y", result.points["y"], heights).max() < plot.tolerance("y")


def test_bars_mode_on_a_crisp_image_is_pixel_exact():
    image = flat_figure((GREEN,))  # one bar: x 70..140, top row 60, down to row 248
    draw = ImageDraw.Draw(image)
    draw.rectangle([200, 100, 259, 248], fill=GREEN)
    draw.rectangle([300, 240, 302, 242], fill=GREEN)  # a speck
    x_axis = AxisCalibration(52, 0.0, 380, 328.0)
    y_axis = AxisCalibration(249.5, 0.0, 49.5, 200.0)  # 1 px = 1 unit, zero at the bottom edge of the bars
    result = digitize(image, x_axis, y_axis, color=GREEN, mode="bars")
    assert result.pixel_points["px"].tolist() == [105.0, 229.5]
    assert result.pixel_points["py"].tolist() == [59.5, 99.5]
    assert result.points["y"].tolist() == pytest.approx([190.0, 150.0])
    assert "1 small speck(s) in the bar colour were ignored" in " | ".join(result.notes)
    limited = digitize(image, x_axis, y_axis, color=GREEN, mode="bars", max_points=1)
    assert limited.n == 1 and "only the first 1 are kept" in " | ".join(limited.notes)


# ====================================================================== errors
def test_digitize_errors():
    plot, _, _ = _points_plot(False, False)
    png = plot.png()
    x_axis, y_axis = plot.calibrations()
    with pytest.raises(DigitizeError, match="Unknown mode"):
        digitize(png, x_axis, y_axis, color=RED, mode="pie")
    with pytest.raises(DigitizeError, match="not a hex colour"):
        digitize(png, x_axis, y_axis, color="red")
    with pytest.raises(DigitizeError, match="No pixels close to #00FF00"):
        digitize(png, x_axis, y_axis, color="#00ff00")
    with pytest.raises(DigitizeError, match="max_points must be at least 1"):
        digitize(png, x_axis, y_axis, color=RED, max_points=0)
    with pytest.raises(DigitizeError, match="plot area is empty"):
        digitize(png, x_axis, y_axis, color=RED, plot_area=(300, 100, 200, 300))
    with pytest.raises(DigitizeError, match="plot_area must be"):
        digitize(png, x_axis, y_axis, color=RED, plot_area=(1, 2, 3))
    with pytest.raises(ValueError, match="not an image"):
        digitize(b"junk", x_axis, y_axis, color=RED)
    # an explicit plot area restricts the search
    left, top, right, bottom = dg.detect_plot_area(png, inner=True)
    half = digitize(png, x_axis, y_axis, color=RED, plot_area=(left, top, (left + right) // 2, bottom))
    assert 0 < half.n < 14 and half.points["x"].max() < 5.2


def test_tolerance_controls_colour_matching():
    image = _dots([(100, 100)], color="#D62728")
    ImageDraw.Draw(image).ellipse([194, 94, 206, 106], fill="#E0393A")  # a slightly different red
    x_axis, y_axis = AxisCalibration(0, 0.0, 400, 400.0), AxisCalibration(300, 0.0, 0, 300.0)
    assert digitize(image, x_axis, y_axis, color=RED, tolerance=2.0).n == 1
    assert digitize(image, x_axis, y_axis, color=RED, tolerance=25.0).n == 2


# ====================================================================== approximate tables and statistics
def _digitized_table():
    plot, xs, ys = _points_plot(False, False, seed=5)
    x_axis, y_axis = plot.calibrations()
    result = digitize(plot.png(), x_axis, y_axis, color=RED, series_name="Response")
    return result, to_table(result, "Figure 2B")


def test_to_table_flags_the_data_as_approximate():
    result, table = _digitized_table()
    assert isinstance(table, XYTable)
    assert table.approximate is True
    assert table.name == "Figure 2B (APPROXIMATE)" and table.name.endswith(" (APPROXIMATE)")
    assert table.notes == "APPROXIMATE: for reference only. Digitized from an image; values are estimates."
    assert table.x_name == "X" and table.series_names == ["Response"]
    np.testing.assert_allclose(table.x, result.points["x"])
    np.testing.assert_allclose(table.series("Response").to_numpy().ravel(), result.points["y"])
    assert "APPROXIMATE" in table.summary() and table.meta()["approximate"] is True
    # default and already-suffixed names
    assert to_table(result).name == "Response (APPROXIMATE)"
    assert to_table(result, "Scan (APPROXIMATE)").name == "Scan (APPROXIMATE)"
    result.series_name = "X"
    assert to_table(result).series_names == ["Y"]
    empty = DigitizeResult(points=result.points.iloc[:0], n=0, mode="points", notes=[],
                           pixel_points=result.pixel_points.iloc[:0])
    with pytest.raises(DigitizeError, match="no digitized points"):
        to_table(empty)


def test_statistics_are_blocked_on_digitized_tables_unless_overridden():
    descriptives = pytest.importorskip("torusplot.stats.descriptives")
    assert descriptives is not None
    from torusplot.stats import ApproximateDataError, run_analysis

    _, table = _digitized_table()
    with pytest.raises(ApproximateDataError, match="APPROXIMATE"):
        run_analysis(table, "descriptives")
    result = run_analysis(table, "descriptives", allow_approximate=True)
    assert result.approximate_data is True
    assert result.warnings and result.warnings[0].startswith("APPROXIMATE DATA")
    assert result.table == table.name
