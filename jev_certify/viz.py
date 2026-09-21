"""Charts: dependency-free SVG generated straight from the recorded results.

Every figure in the README is produced by this module from ``results/results.json``, so no
chart can show a number the analysis did not measure. SVGs are written as files (rather
than inline HTML) because GitHub renders committed ``.svg`` images and strips inline SVG
from Markdown.

The charts use explicit light colours instead of theme variables: they are committed
artifacts rendered inside someone else's page, so they carry their own background and stay
legible on both GitHub light and dark themes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
PAPER = "#ffffff"
BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
VIOLET = "#7c3aed"
GRAY = "#9ca3af"

FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


# ------------------------------------------------------------------------- primitives


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _frame(width: int, height: int, title: str, subtitle: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{_esc(title)}">
<rect width="{width}" height="{height}" fill="{PAPER}"/>
<text x="24" y="30" font-family="{FONT}" font-size="16" font-weight="600" fill="{INK}">{_esc(title)}</text>
<text x="24" y="50" font-family="{FONT}" font-size="11.5" fill="{MUTED}">{_esc(subtitle)}</text>
{body}
</svg>
"""


class Plot:
    """Cartesian area with helpers for gridlines, ticks and coordinates."""

    def __init__(
        self,
        width: int,
        height: int,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        y_log: bool = False,
        pad_left: int = 82,
        pad_right: int = 28,
        pad_top: int = 78,
        pad_bottom: int = 58,
    ):
        self.width, self.height = width, height
        self.xlim, self.ylim = xlim, ylim
        self.y_log = y_log
        self.pl, self.pr, self.pt, self.pb = pad_left, pad_right, pad_top, pad_bottom

    @property
    def plot_w(self) -> float:
        return self.width - self.pl - self.pr

    @property
    def plot_h(self) -> float:
        return self.height - self.pt - self.pb

    def sx(self, x: float) -> float:
        lo, hi = self.xlim
        return self.pl + (x - lo) / (hi - lo or 1.0) * self.plot_w

    def sy(self, y: float) -> float:
        lo, hi = self.ylim
        if self.y_log:
            lo, hi = max(lo, 1e-9), max(hi, 1e-9)
            y = max(y, 1e-9)
            frac = (math.log10(y) - math.log10(lo)) / (math.log10(hi) - math.log10(lo) or 1.0)
        else:
            frac = (y - lo) / (hi - lo or 1.0)
        return self.pt + self.plot_h - frac * self.plot_h

    def axes(self, x_label: str, y_label: str, x_ticks: Sequence[float], y_ticks: Sequence[float]) -> str:
        parts = [
            f'<rect x="{self.pl}" y="{self.pt}" width="{self.plot_w}" height="{self.plot_h}" '
            f'fill="none" stroke="{GRID}"/>'
        ]
        for value in y_ticks:
            y = self.sy(value)
            parts.append(
                f'<line x1="{self.pl}" y1="{y:.1f}" x2="{self.pl + self.plot_w:.1f}" y2="{y:.1f}" '
                f'stroke="{GRID}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{self.pl - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="{FONT}" '
                f'font-size="10.5" fill="{MUTED}">{_esc(_tick(value, self.y_log))}</text>'
            )
        for value in x_ticks:
            x = self.sx(value)
            parts.append(
                f'<text x="{x:.1f}" y="{self.pt + self.plot_h + 18:.1f}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="10.5" fill="{MUTED}">{_esc(_tick(value, False))}</text>'
            )
        parts.append(
            f'<text x="{self.pl + self.plot_w / 2:.1f}" y="{self.height - 16}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11.5" fill="{INK}">{_esc(x_label)}</text>'
        )
        parts.append(
            f'<text x="18" y="{self.pt + self.plot_h / 2:.1f}" font-family="{FONT}" font-size="11.5" '
            f'fill="{INK}" transform="rotate(-90 18 {self.pt + self.plot_h / 2:.1f})" '
            f'text-anchor="middle">{_esc(y_label)}</text>'
        )
        return "".join(parts)


def _tick(value: float, is_log: bool) -> str:
    if is_log:
        return f"{value:g}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:g}"
    return f"{value:.2f}".rstrip("0").rstrip(".") if value else "0"


def _log_ticks(ylim: tuple[float, float]) -> list[float]:
    low, high = math.log10(ylim[0]), math.log10(ylim[1])
    ticks = []
    for decade in range(math.floor(low), math.ceil(high) + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * 10**decade
            if ylim[0] <= value <= ylim[1]:
                ticks.append(value)
    return ticks


def line_chart(
    title: str,
    subtitle: str,
    series: Sequence[dict[str, Any]],
    x_label: str,
    y_label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    y_log: bool = False,
    width: int = 900,
    height: int = 520,
) -> str:
    """Line/scatter chart. Each series: {name, points: [(x, y)], color, dashed?}"""
    plot = Plot(width, height, xlim, ylim, y_log=y_log)
    y_ticks = _log_ticks(ylim) if y_log else [ylim[0] + (ylim[1] - ylim[0]) * i / 4 for i in range(5)]
    x_ticks = [xlim[0] + (xlim[1] - xlim[0]) * i / 5 for i in range(6)]
    body = [plot.axes(x_label, y_label, x_ticks, y_ticks)]

    for index, item in enumerate(series):
        points = [(float(x), float(y)) for x, y in item["points"]]
        if not points:
            continue
        colour = item.get("color", BLUE)
        dash = ' stroke-dasharray="6 4"' if item.get("dashed") else ""
        if item.get("line", True) and len(points) > 1:
            path = " ".join(
                f"{'M' if i == 0 else 'L'}{plot.sx(x):.1f},{plot.sy(y):.1f}" for i, (x, y) in enumerate(points)
            )
            body.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.2"{dash}/>')
        for x, y in points:
            body.append(
                f'<circle cx="{plot.sx(x):.1f}" cy="{plot.sy(y):.1f}" r="4.2" fill="{colour}" '
                f'stroke="{PAPER}" stroke-width="1.4"/>'
            )
        # Legend, laid out in two columns so eight series still fit.
        column, row = divmod(index, 4)
        lx = plot.pl + 8 + column * 300
        ly = plot.pt + 16 + row * 18
        body.append(
            f'<rect x="{lx}" y="{ly - 8}" width="16" height="3" fill="{colour}" rx="1.5"/>'
            f'<circle cx="{lx + 8}" cy="{ly - 6.5}" r="3.4" fill="{colour}"/>'
            f'<text x="{lx + 24}" y="{ly - 2}" font-family="{FONT}" font-size="11" fill="{INK}">'
            f'{_esc(item["name"])}</text>'
        )
    return _frame(width, height, title, subtitle, "".join(body))


def _wrap(label: str, max_chars: int, max_lines: int = 3) -> list[str]:
    """Greedy word wrap, so axis labels stay inside their band instead of colliding."""
    words = str(label).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                current = ""
                break
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:max_lines] or [str(label)]


def grouped_bars(
    title: str,
    subtitle: str,
    categories: Sequence[str],
    groups: Sequence[dict[str, Any]],
    y_label: str,
    ylim: tuple[float, float] | None = None,
    y_log: bool = False,
    width: int = 900,
    height: int = 520,
    value_fmt: Callable[[float], str] = lambda v: f"{v:.4f}",
    annotate: bool = True,
) -> str:
    """Grouped bar chart with optional log scale (risks span three orders of magnitude)."""
    values = [v for group in groups for v in group["values"] if v is not None]
    top = max(values) if values else 1.0
    if ylim is None:
        ylim = (min(values) * 0.5 if y_log and values else 0.0, top * 1.25)
    plot = Plot(width, height, (0, len(categories)), ylim, y_log=y_log, pad_bottom=50 + 14 * 2)
    y_ticks = _log_ticks(ylim) if y_log else [ylim[0] + (ylim[1] - ylim[0]) * i / 4 for i in range(5)]
    # No numeric x ticks: the categories own the x axis, and drawing both collides.
    body = [plot.axes("", y_label, [], y_ticks)]

    band = plot.plot_w / max(1, len(categories))
    max_chars = max(8, int(band / 6.4))
    for index, label in enumerate(categories):
        x = plot.sx(index + 0.5)
        for offset, text in enumerate(_wrap(label, max_chars)):
            body.append(
                f'<text x="{x:.1f}" y="{plot.pt + plot.plot_h + 20 + offset * 13:.1f}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="11" fill="{INK}">{_esc(text)}</text>'
            )

    bar_w = min(46.0, band * 0.78 / max(1, len(groups)))
    for index, group in enumerate(groups):
        colour = group.get("color", BLUE)
        for c_index, value in enumerate(group["values"]):
            if value is None:
                continue
            centre = plot.pl + band * (c_index + 0.5)
            offset = (index - (len(groups) - 1) / 2) * (bar_w + 3)
            x = centre + offset - bar_w / 2
            y = plot.sy(value)
            base = plot.sy(ylim[0])
            height_px = max(1.5, base - y)
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height_px:.1f}" rx="2.5" '
                f'fill="{colour}" opacity="0.9"/>'
            )
            if annotate:
                body.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
                    f'font-family="{FONT}" font-size="10" fill="{MUTED}">{_esc(value_fmt(value))}</text>'
                )
    for index, group in enumerate(groups):
        lx = plot.pl + 8 + index * 250
        ly = plot.pt + 16
        body.append(
            f'<rect x="{lx}" y="{ly - 9}" width="14" height="9" rx="2" fill="{group.get("color", BLUE)}"/>'
            f'<text x="{lx + 20}" y="{ly - 2}" font-family="{FONT}" font-size="11" fill="{INK}">'
            f'{_esc(group["name"])}</text>'
        )
    return _frame(width, height, title, subtitle, "".join(body))


def histogram(
    title: str,
    subtitle: str,
    counts: Sequence[int],
    errors: Sequence[int],
    bin_width: float,
    x_label: str,
    y_label: str,
    width: int = 900,
    height: int = 500,
) -> str:
    """Stacked histogram: total answers per bin, with the wrong ones highlighted."""
    top = max(counts) if counts else 1
    span = len(counts) * bin_width
    plot = Plot(width, height, (0.0, span), (0, top * 1.15))
    # The x axis is probability, not bin index -- a reader should not have to convert.
    x_ticks = [span * i / 5 for i in range(6)]
    y_ticks = [top * 1.15 * i / 5 for i in range(6)]
    body = [plot.axes(x_label, y_label, x_ticks, y_ticks)]

    band = plot.sx(bin_width) - plot.sx(0.0)
    for index, count in enumerate(counts):
        x = plot.sx(index * bin_width) + band * 0.1
        wide = band * 0.8
        base = plot.sy(0)
        y_total = plot.sy(count)
        body.append(
            f'<rect x="{x:.1f}" y="{y_total:.1f}" width="{wide:.1f}" height="{max(0.8, base - y_total):.1f}" '
            f'rx="2" fill="{BLUE}" opacity="0.55"/>'
        )
        wrong = min(errors[index], count)
        if wrong:
            y_wrong = plot.sy(wrong)
            body.append(
                f'<rect x="{x:.1f}" y="{y_wrong:.1f}" width="{wide:.1f}" height="{max(0.8, base - y_wrong):.1f}" '
                f'rx="2" fill="{RED}"/>'
            )
    body.append(
        f'<rect x="{plot.pl + 8}" y="{plot.pt + 8}" width="14" height="9" rx="2" fill="{BLUE}" opacity="0.55"/>'
        f'<text x="{plot.pl + 28}" y="{plot.pt + 16}" font-family="{FONT}" font-size="11" fill="{INK}">'
        "answers</text>"
        f'<rect x="{plot.pl + 108}" y="{plot.pt + 8}" width="14" height="9" rx="2" fill="{RED}"/>'
        f'<text x="{plot.pl + 128}" y="{plot.pt + 16}" font-family="{FONT}" font-size="11" fill="{INK}">'
        "of which wrong</text>"
    )
    return _frame(width, height, title, subtitle, "".join(body))


def reliability_chart(
    title: str,
    subtitle: str,
    bins: Sequence[dict[str, Any]],
    width: int = 900,
    height: int = 500,
) -> str:
    """Reliability diagram with the y = x diagonal a calibrated model should follow."""
    plot = Plot(width, height, (0, 1), (0, 1))
    ticks = [i / 5 for i in range(6)]
    body = [plot.axes("Jev's probability of 'in scope'", "observed frequency", ticks, ticks)]
    body.append(
        f'<line x1="{plot.sx(0):.1f}" y1="{plot.sy(0):.1f}" x2="{plot.sx(1):.1f}" y2="{plot.sy(1):.1f}" '
        f'stroke="{GRAY}" stroke-dasharray="6 4"/>'
    )
    body.append(
        f'<text x="{plot.sx(0.72):.1f}" y="{plot.sy(0.80):.1f}" font-family="{FONT}" font-size="11" '
        f'fill="{MUTED}">perfect calibration</text>'
    )
    points = [(b["mean_confidence"], b["empirical_frequency"]) for b in bins]
    if points:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{plot.sx(x):.1f},{plot.sy(y):.1f}" for i, (x, y) in enumerate(points)
        )
        body.append(f'<path d="{path}" fill="none" stroke="{BLUE}" stroke-width="2.2"/>')
        biggest = max(b["count"] for b in bins)
        for (x, y), source in zip(points, bins):
            radius = 4 + 8 * (source["count"] / biggest) ** 0.5
            body.append(
                f'<circle cx="{plot.sx(x):.1f}" cy="{plot.sy(y):.1f}" r="{radius:.1f}" fill="{BLUE}" '
                f'opacity="0.75" stroke="{PAPER}" stroke-width="1.2"/>'
            )
    body.append(
        f'<text x="{plot.pl + 8}" y="{plot.pt + plot.plot_h - 10}" font-family="{FONT}" font-size="11" '
        f'fill="{MUTED}">point size = answers in that bin</text>'
    )
    return _frame(width, height, title, subtitle, "".join(body))


# ------------------------------------------------------------------------ figure set


def write_charts(result: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Render every README figure from the analysis result. Returns the files written."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    routing = result["routing"]
    gate = result["gate"]
    audit = result["audit"]
    drift = result["drift"]
    points = routing.get("operating_points") or {}
    alpha = result["meta"]["alpha"]

    def save(name: str, svg: str) -> None:
        path = target / name
        path.write_text(svg, encoding="utf-8")
        written.append(path)

    # 1. Risk-coverage, with the certified and hand-picked operating points on it.
    curve = routing["curves"]["in_scope_only"]
    certified_series = {
        "name": "certified operating points (α = 2/5/10%)",
        "points": [],
        "color": GREEN,
        "line": False,
    }
    for cert in routing["certificates"]:
        if cert["feasible"]:
            certified_series["points"].append((cert["coverage"], cert["measured_on_holdout"]["per_query_risk"]))
    hand = {
        "name": "hand-picked thresholds",
        "points": [
            (p["coverage"], p["per_query_loss"])
            for p in points.get("operating_points", [])
            if "hand-picked" in p["name"]
        ],
        "color": AMBER,
        "line": False,
    }
    full_curve = {
        "name": "measured risk-coverage",
        "points": [(row["coverage"], row["per_query_risk"]) for row in curve],
        "color": BLUE,
    }
    save(
        "risk-coverage.svg",
        line_chart(
            "Certified routing threshold vs. hand-picked ones",
            f"Jev 1.13 on CLINC150 · {routing['evaluation_n']} held-out human-labelled queries · "
            "loss = share of incoming queries silently misrouted",
            [full_curve, certified_series, hand],
            "coverage (share of traffic auto-routed)",
            "loss / query",
            xlim=(0.55, 1.0),
            ylim=(0.001, 0.12),
            y_log=True,
        ),
    )

    # 2. The audit: interval width vs labelling budget, by estimator.
    budgets = audit["budgets"]
    save(
        "ppi-audit.svg",
        line_chart(
            "Auditing the live policy: how many hand labels do you actually need?",
            f"prediction-powered inference on {audit['pool_n']} decisions · 500 Monte-Carlo draws per budget "
            "· 95% intervals",
            [
                {"name": "labels only (classical)", "points": [(b["n_labeled"], b["methods"]["classical"]["mean_width"]) for b in budgets], "color": RED},
                {"name": "PPI++ (labels + Jev)", "points": [(b["n_labeled"], b["methods"]["ppi"]["mean_width"]) for b in budgets], "color": GREEN},
                {"name": "Jev only (imputation)", "points": [(b["n_labeled"], b["methods"]["imputation"]["mean_width"]) for b in budgets], "color": GRAY},
            ],
            "hand-labelled decisions",
            "95% interval width (lower is better)",
            xlim=(0, max(b["n_labeled"] for b in budgets) * 1.08),
            ylim=(0.0, max(b["methods"]["classical"]["mean_width"] for b in budgets) * 1.1),
        ),
    )

    # 3. The prevalence break: certified vs measured, same mixer vs production mixer.
    certificates = [c for c in gate["certificates"] if c.get("feasible")]
    if certificates:
        save(
            "prevalence.svg",
            grouped_bars(
                "The scope gate's certificate is prevalence-sensitive",
                f"certified false-accepts/query vs. measured on two traffic mixes · calibration mix "
                f"{100 * gate['calibration_uncovered_prevalence']:.1f}% uncovered, evaluation mix "
                f"{100 * gate['evaluation_uncovered_prevalence']:.1f}% uncovered",
                [f"α = {c['alpha']:.0%}" for c in certificates],
                [
                    {
                        "name": "certified bound",
                        "values": [c["certified_false_accept_per_query"] for c in certificates],
                        "color": GRAY,
                    },
                    {
                        "name": "measured, matched prevalence",
                        "values": [
                            (c.get("measured_on_prevalence_matched_mix") or {}).get("per_query_risk")
                            for c in certificates
                        ],
                        "color": BLUE,
                    },
                    {
                        "name": "measured, production prevalence",
                        "values": [(c.get("measured") or {}).get("per_query_risk") for c in certificates],
                        "color": RED,
                    },
                ],
                "loss / query",
                ylim=(0.0, max((c.get("measured") or {}).get("per_query_risk") or 0 for c in certificates) * 1.25),
                value_fmt=lambda v: f"{v:.4f}",
            ),
        )

    # 4. What the certificate buys, and where it dies. Labels are written for a reader
    # rather than echoing the internal section keys.
    friendly = {
        "evaluation_exchangeable": "held-out mix, as calibrated",
        "in_scope_taxonomy_exchangeable": "in-scope only, as calibrated",
        "in_taxonomy_exchangeable": "in-scope only, as calibrated",
        "out_of_scope_traffic": "out-of-scope traffic arrives",
        "unlisted_intents_traffic": "restricted deployment: unlisted intents",
    }
    populations: list[tuple[str, float, bool]] = []
    for name, section in drift["sections"].items():
        populations.append(
            (
                friendly.get(name, name.replace("_", " ")),
                section["routing_above_threshold"]["per_query_risk"],
                section["certificate_held"],
            )
        )
    restricted = result.get("restricted_deployment")
    if restricted:
        populations.append(
            (
                friendly["unlisted_intents_traffic"],
                restricted["drift"]["sections"]["unlisted_intents_traffic"]["routing_above_threshold"][
                    "per_query_risk"
                ],
                False,
            )
        )
    if populations:
        save(
            "where-it-breaks.svg",
            grouped_bars(
                "Where the certificate holds, and where it dies",
                f"the α = {alpha:.0%} certificate applied to traffic it was never calibrated on · same model, "
                "same thresholds, same code",
                [name for name, _, _ in populations],
                [
                    {
                        "name": "certified bound",
                        "values": [alpha] * len(populations),
                        "color": GRAY,
                    },
                    {
                        "name": "measured loss / query",
                        "values": [value for _, value, _ in populations],
                        "color": RED,
                    },
                ],
                "loss / query (log scale)",
                ylim=(0.001, 1.6),
                y_log=True,
                value_fmt=lambda v: f"{v:.4f}",
            ),
        )

    # 5. The resolution floor: the distribution of Jev's top probability.
    quantization = result.get("quantization")
    if quantization and quantization.get("histogram_counts"):
        share = 100 * (quantization.get("share_at_1.0") or 0)
        errors_at_one = quantization.get("errors_at_1.0") or 0
        save(
            "resolution-floor.svg",
            histogram(
                "Why a 1% risk target is unreachable: Jev's own probability rounding",
                f"{quantization['answers']} routed answers · {share:.1f}% come back at exactly 1.0 and "
                f"{errors_at_one} of those are wrong, so no threshold can separate them",
                quantization["histogram_counts"],
                quantization["histogram_errors"],
                quantization["histogram_bin_width"],
                "maximum probability Jev returned for the query",
                "answers",
            ),
        )

    # 6. Reliability of the gate's probability against human labels.
    reliability = (gate.get("noul_diagnostics") or {}).get("reliability") or []
    if reliability:
        save(
            "reliability.svg",
            reliability_chart(
                "Does Jev's probability mean what it says?",
                f"scope question (`noul`) vs. human labels on the calibration split · "
                f"ECE = {gate['noul_diagnostics']['ece10']:.3f}, Brier = {gate['noul_diagnostics']['brier']:.3f}",
                reliability,
            ),
        )
    return written
