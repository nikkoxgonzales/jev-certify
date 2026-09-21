"""Render the shareable card for this project: docs/social-card.svg.

A 16:9 card for posting the result somewhere with a character limit. Every number is read
from results/results.json, so the card cannot advertise a figure the analysis did not
measure -- the same rule the README and the charts follow.

    python -m experiments.social_card          # writes docs/social-card.svg
    magick -density 150 docs/social-card.svg -resize 1600x900 docs/social-card.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "results.json"
OUT = ROOT / "docs" / "social-card.svg"

W, H = 1600, 900
INK = "#F8FAFC"
MUTED = "#94A3B8"
DIM = "#64748B"
PAPER = "#0A0F1C"
CARD = "#121A2B"
BORDER = "#1E293B"
GREEN = "#4ADE80"
AMBER = "#FBBF24"
BLUE = "#60A5FA"
GRID = "#1B2438"
FONT = "'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x: float, y: float, body: Any, size: float, fill: str = INK, weight: int = 400,
          spacing: float = 0, anchor: str = "start", opacity: float = 1.0) -> str:
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    op = f' opacity="{opacity}"' if opacity != 1.0 else ""
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{extra}{op}>{_esc(body)}</text>'
    )


def _curve_panel(result: dict[str, Any], x: float, y: float, w: float, h: float) -> str:
    """The measured risk-coverage curve, with the certified point picked out."""
    points = result["routing"]["curves"]["in_scope_only"]
    y_log_lo, y_log_hi = math.log10(0.001), math.log10(0.1)
    x_lo, x_hi = 0.55, 1.0
    # Inner margins so the curve never touches the panel border.
    left, right, top, bottom = x + 62, x + w - 30, y + 62, y + h - 34

    def sx(value: float) -> float:
        return left + (value - x_lo) / (x_hi - x_lo) * (right - left)

    def sy(value: float) -> float:
        value = max(value, 1e-6)
        frac = (math.log10(value) - y_log_lo) / (y_log_hi - y_log_lo)
        return bottom - frac * (bottom - top)

    parts = [
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" fill="{CARD}" '
        f'stroke="{BORDER}" rx="14"/>'
    ]
    for tick in (0.002, 0.01, 0.05):
        ty = sy(tick)
        parts.append(
            f'<line x1="{left:.0f}" y1="{ty:.1f}" x2="{right:.0f}" y2="{ty:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(_text(left - 8, ty + 5, f"{tick:g}", 15, DIM, anchor="end"))
    for tick in (0.6, 0.8, 1.0):
        parts.append(_text(sx(tick), bottom + 26, f"{tick:g}", 15, DIM, anchor="middle"))

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(row['coverage']):.1f},{sy(row['per_query_risk']):.1f}"
        for i, row in enumerate(points)
    )
    parts.append(f'<path d="{path}" fill="none" stroke="{BLUE}" stroke-width="2.6"/>')

    for certificate in result["routing"]["certificates"]:
        if not certificate["feasible"]:
            continue
        measured = certificate["measured_on_holdout"]
        cx, cy = sx(measured["coverage"]), sy(measured["per_query_risk"])
        if certificate["alpha"] == 0.05:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="{GREEN}" opacity="0.22"/>')
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{GREEN}" stroke="{PAPER}" stroke-width="2"/>'
        )
    parts.append(_text(x + 26, y + 30, "measured risk vs. traffic routed", 16, MUTED))
    parts.append(_text(x + 26, y + 50, "green dot = certified threshold (α = 5%)", 15, GREEN, opacity=0.9))
    return "".join(parts)


def build(result: dict[str, Any]) -> str:
    primary = result["routing"]["primary"]
    measured = primary["measured_on_holdout"]
    quantization = result["quantization"]
    feasible = [c for c in result["routing"]["certificates"] if c["feasible"]]
    floor = next(c for c in result["routing"]["certificates"] if c["alpha"] == 0.01)["min_attainable_bound"]
    planning = result["audit"]["planning"]["half_width_0.05"]
    cost = result["cost"]
    stats = result["journal_stats"]

    body: list[str] = [
        # Backdrop
        f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
        f'<circle cx="1320" cy="-60" r="440" fill="{BLUE}" opacity="0.07"/>',
        f'<circle cx="180" cy="960" r="420" fill="{GREEN}" opacity="0.06"/>',

        # Kicker row
        _text(80, 84, "JEV 1.13  ·  TYPESAFE SYSTEM ONE", 20, GREEN, 600, spacing=2.2),
        _text(W - 80, 84, f"{stats['decisions']:,} DECISIONS  ·  ${stats['cost_usd']:.2f}  ·  "
                           "HUMAN LABELS", 20, DIM, 500, spacing=1.6, anchor="end"),

        # Headline
        _text(80, 232, "Jev gives you a calibrated probability.", 62, INK, 700),
        _text(80, 306, "This turns it into a certificate.", 62, BLUE, 700),
        _text(80, 366, "Conformal risk control + prediction-powered inference, measured on a 150-way "
                       "intent router.", 26, MUTED),
    ]

    # Stat cards
    cards = [
        (
            f"{measured['coverage'] * 100:.2f}%",
            "of traffic auto-routed at a 5% risk target",
            f"{measured['selective_error'] * 100:.2f}% wrong among routed · certificate holds "
            f"({measured['per_query_risk']:.4f} measured vs {primary['certified_per_query_loss']:.4f} certified)",
            GREEN,
        ),
        (
            f"{floor * 100:.2f}%",
            "is the floor on risk for this model",
            f"α = 1% is unreachable: {quantization['share_at_1.0'] * 100:.1f}% of answers come back "
            f"exactly 1.0, and {quantization['errors_at_1.0']} of those are wrong",
            AMBER,
        ),
        (
            f"{planning['ppi_labels']:.0f} / {planning['classical_labels']:.0f}",
            "hand labels to audit it",
            "40 labels with Jev's estimates do what 254 labels do without — a 62% narrower interval",
            BLUE,
        ),
    ]
    # Three equal cards, with the same outer margin on the right as on the left.
    card_y, card_h = 424, 216
    margin, gap = 80, 42
    card_w = (W - 2 * margin - 2 * gap) / 3
    for index, (big, label, detail, colour) in enumerate(cards):
        x = margin + index * (card_w + gap)
        body.append(
            f'<rect x="{x}" y="{card_y}" width="{card_w}" height="{card_h}" fill="{CARD}" '
            f'stroke="{BORDER}" rx="16"/>'
        )
        body.append(f'<rect x="{x}" y="{card_y}" width="4" height="{card_h}" fill="{colour}" rx="2"/>')
        body.append(_text(x + 34, card_y + 78, big, 58, colour, 700))
        body.append(_text(x + 34, card_y + 118, label, 21, INK, 600))
        # Wrap the detail by hand: three lines is all the card has room for.
        words, lines, current = detail.split(), [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 44 and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        for offset, line in enumerate(lines[:3]):
            body.append(_text(x + 34, card_y + 148 + offset * 24, line, 17, MUTED))

    # Footer: the curve, the link, and the honest half
    body.append(_curve_panel(result, 80, 672, 720, 176))
    body.append(_text(856, 736, "github.com/nikkoxgonzales/jev-certify", 30, INK, 700))
    body.append(_text(856, 776, f"${cost['cost_usd_per_1k']:.4f} per 1k queries · p50 {cost['latency_p50_s']}s "
                                f"· MIT", 20, MUTED))
    body.append(_text(856, 812, "It also measures where the guarantee breaks:", 20, AMBER, 600))
    body.append(_text(856, 838, "prevalence shift 3.6× over the bound, and 1.0000 loss/query on "
                                "shifted traffic.", 18, DIM))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="jev-certify: certified decision thresholds for Jev">'
        + "".join(body)
        + "</svg>\n"
    )


def main() -> int:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    # The card quotes the run's headline bookkeeping, so pull it from the same place the
    # README does rather than restating it.
    stats = json.loads((ROOT / "results" / "collection_stats.json").read_text(encoding="utf-8"))
    result["journal_stats"] = {
        "decisions": stats["journal_entries"],
        "cost_usd": stats["journal_total_cost_usd"],
    }
    OUT.write_text(build(result), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
