"""Deterministic technical flats (front and back view) drawn from style categories.

The flat shows what SeamProof will *draft*, not what was asked for: features
the pattern library cannot draft are swapped for what it can (see
`drafted_features`), so the sketch never promises a detail the pattern lacks.
No model ever draws here; coordinates are fixed illustration units, not mm.
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape

from seamproof.style import SKIRT_FLARE, SKIRT_LENGTH, StylePlan

# Named colours a design may use; Muse picks a key, never a hex value.
COLORS: dict[str, tuple[str, str]] = {
    "putih": ("#F6F5F0", "White"), "broken_white": ("#EDE8DC", "Broken white"),
    "krem": ("#E9DFC9", "Cream"), "mocca": ("#A47E62", "Mocca"), "coklat": ("#7A5A43", "Brown"),
    "hitam": ("#2A2A2A", "Black"), "abu": ("#9A9C9A", "Grey"), "navy": ("#2F3E5C", "Navy"),
    "biru": ("#6E93C2", "Blue"), "sage": ("#9DB09A", "Sage"), "army": ("#5B5E3E", "Army"),
    "emerald": ("#2F7D62", "Emerald"), "maroon": ("#6E2B32", "Maroon"), "merah": ("#B5433A", "Red"),
    "terracotta": ("#B5674A", "Terracotta"), "salem": ("#E9A68E", "Salem"),
    "dusty_pink": ("#D3A6A4", "Dusty pink"), "lilac": ("#B7A6CC", "Lilac"),
    "mustard": ("#C9A03A", "Mustard"), "kuning": ("#E6CC6A", "Yellow"),
}
DEFAULT_COLOR = "broken_white"

# What each standard block looks like; their style is fixed by the block.
BLOCK_LOOKS: dict[str, dict[str, Any]] = {
    "tshirt": {"collar": "none", "opening": "none", "sleeve_length": "short", "shoulder": "set_in",
               "fit": "regular", "length": "hip", "pockets": 0, "cuff": False},
}

# Illustration units: each view is VIEW_W wide (long sleeves reach about 250).
VIEW_W = 270.0
VIEW_PAD = 15.0
CX = 120.0
SHOULDER_Y = 40.0
ARMPIT_Y = 84.0
BODY_HALF = {"fitted": 44.0, "regular": 50.0, "boxy": 60.0}
TOP_HEM_Y = {"waist": 140.0, "high_hip": 158.0, "hip": 178.0, "thigh": 204.0, "knee": 240.0,
             "below_knee": 262.0}
SLEEVE_LEN = {"sleeveless": 0.0, "short": 34.0, "elbow": 62.0, "three_quarter": 96.0, "long": 134.0}
SKIRT_HEM_Y = {"mini": 140.0, "above_knee": 165.0, "knee": 195.0, "midi": 235.0, "maxi": 272.0}
SKIRT_HEM_HALF = {"straight": 54.0, "a_line": 70.0, "flared": 94.0}
DRESS_WAIST_Y = 136.0
DRESS_HEM_Y = {"mini": 200.0, "above_knee": 226.0, "knee": 252.0, "midi": 282.0, "maxi": 308.0}
# View height per part: a dress runs longest (maxi hem at 308).
VIEW_H = {"top": 290.0, "bottom": 300.0, "dress": 330.0}


def color_hex(name: str | None) -> str:
    """Hex for a known colour key; unknown keys fall back to broken white."""
    return COLORS.get(name or "", COLORS[DEFAULT_COLOR])[0]


def _luminance(hex_color: str) -> float:
    """Relative brightness 0..1 of a #RRGGBB colour (simple average weighting)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def drafted_features(plan: StylePlan) -> dict[str, Any]:
    """Features as the pattern library will actually draft them.

    Mirrors the substitutions in seamproof.style (and its notes): shirts
    always open at the front; sleeveless shirts have a fitted shoulder and no
    cuff; band and unknown collars become pointed and peter pan; a V neck
    needs no collar; skirts fall back to A-line and a straight hem. Blouse and
    tunic are collarless with a set-in shoulder and their own lengths; the
    tshirt block ignores photo style.
    """
    f = dict(plan.features)
    if plan.garment in BLOCK_LOOKS:
        return dict(BLOCK_LOOKS[plan.garment])
    if plan.garment in ("blouse", "tunic"):
        f.update(collar="none", opening="none", shoulder="set_in", pockets=0, contrast_collar=False)
        f.setdefault("sleeve_length", "long")
        f.setdefault("fit", "fitted")
        if plan.garment == "blouse":
            f.update(length="hip", fit="fitted")
        else:
            f.setdefault("length", "knee")
            if f["length"] in ("waist", "high_hip"):
                f["length"] = "hip"
    if plan.garment in ("shirt", "blouse", "tunic"):
        if plan.garment == "shirt":
            f["opening"] = "button_front"
        if plan.kind == "vest":
            f["sleeve_length"] = "sleeveless"
        # Mirrors style._cuffed_sleeve: bishop/puff sleeves always gather into a cuff, elastic at the wrist.
        if f.get("sleeve_style", "straight") != "straight":
            f["cuff"] = True
            if f.get("sleeve_length") == "long":
                f["elastic_cuff"] = True
        f["elastic_cuff"] = bool(f.get("elastic_cuff")) and bool(f.get("cuff") or f.get("elastic_cuff"))
        if f.get("elastic_cuff"):
            f["cuff"] = True
        if f.get("sleeve_length") == "sleeveless":
            f.update(shoulder="set_in", cuff=False, elastic_cuff=False, sleeve_style="straight")
        f["collar"] = {"other": "peter_pan"}.get(f.get("collar", "none"), f.get("collar", "none"))
        if f["collar"] != "none":
            f["neckline"] = "round"
        if plan.garment == "shirt" and f.get("length") == "below_knee":
            f["length"] = "knee"
    elif plan.garment in ("skirt", "dress"):
        if f.get("silhouette") not in SKIRT_HEM_HALF:
            f["silhouette"] = "a_line"
        if f.get("hem") != "scalloped" or (plan.garment == "dress" and not f.get("waist_seam")):
            f["hem"] = "straight"
    if plan.garment == "dress":
        # Mirrors style._dress: plain neck with facing, side zip, hemmed sleeves, set-in shoulder.
        f.update(collar="none", opening="none", cuff=False, elastic_cuff=False, sleeve_style="straight",
                 pockets=0, shoulder="set_in", fit="regular")
    if plan.garment not in ("shirt", "dress", "blouse", "tunic"):
        f["neckline"] = "round"
    return f


def _lerp(x: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear lookup that extrapolates past both ends."""
    i = 1 if x < xs[1] else len(xs) - 1
    for j in range(1, len(xs)):
        if x <= xs[j]:
            i = j
            break
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])


def view_features(plan: StylePlan) -> dict[str, Any]:
    """Drafted features plus fitted hem size in illustration units, for flat and 3D.

    The same category tables are interpolated, so a fitted value equal to a
    category's value draws exactly that category.
    """
    f = drafted_features(plan)
    if plan.proportions:
        flares = sorted((SKIRT_FLARE[k], SKIRT_HEM_HALF[k]) for k in SKIRT_HEM_HALF)
        hem_y = DRESS_HEM_Y if plan.garment == "dress" else SKIRT_HEM_Y
        f["hem_half"] = _lerp(plan.params["skirt_flare"], [a for a, _ in flares], [b for _, b in flares])
        f["hem_y"] = _lerp(plan.params["skirt_length"], [SKIRT_LENGTH[k] for k in hem_y], list(hem_y.values()))
    return f


def hem_size(f: dict[str, Any], table: dict[str, float], default_y: float) -> tuple[float, float]:
    """(hem half width, hem y): fitted values when present, otherwise the category tables."""
    return (f.get("hem_half", SKIRT_HEM_HALF.get(f.get("silhouette", "a_line"), 70.0)),
            f.get("hem_y", table.get(f.get("length", "knee"), default_y)))


def _neck_shape(f: dict[str, Any], back: bool) -> tuple[float, float | None]:
    """Neck half width and, for a shaped front neckline, its curve depth (None keeps the default)."""
    shape = f.get("neckline") if f.get("collar", "none") == "none" else None
    half = 24.0 if shape == "boat" else 17.0
    return half, None if back else {"scoop": 32.0, "boat": 7.0}.get(shape or "")


def _neck(f: dict[str, Any], back: bool, neck_half: float, neck_y: float, depth: float) -> str:
    """Neckline path: a curve, or straight lines to a V point on the front."""
    if not back and f.get("neckline") == "v" and f.get("collar", "none") == "none":
        return f'M{CX - neck_half:.1f},{neck_y:.1f} L{CX:.1f},{neck_y + 44:.1f} L{CX + neck_half:.1f},{neck_y:.1f}'
    return f'M{CX - neck_half:.1f},{neck_y:.1f} Q{CX:.1f},{neck_y + depth * 1.6:.1f} {CX + neck_half:.1f},{neck_y:.1f}'


def _armhole_line(shoulder_x: float, shoulder_y: float, half: float) -> str:
    """Bound armhole curve drawn where a sleeve would be."""
    return (f'<path class="detail" d="M{shoulder_x:.1f},{shoulder_y:.1f} Q{CX + half - 14:.1f},{ARMPIT_Y - 18:.1f} '
            f'{CX + half:.1f},{ARMPIT_Y:.1f}"/>')


def _pts(points: list[tuple[float, float]]) -> str:
    """SVG points attribute."""
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _mirror(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Reflect points across the centre line."""
    return [(2 * CX - x, y) for x, y in points]


def _sleeve(f: dict[str, Any], half: float, shoulder_x: float, shoulder_y: float) -> tuple[list, list]:
    """Right sleeve outline and its cuff band (empty when sleeveless or no cuff)."""
    length = SLEEVE_LEN.get(f.get("sleeve_length", "short"), 34.0)
    if length <= 0:
        return [], []
    dx, dy = 0.55, 0.835  # arm hangs about 57 degrees below horizontal
    width = ARMPIT_Y - shoulder_y + 6
    end_w = max(16.0, width * (0.95 - 0.004 * length))
    top_end = (shoulder_x + dx * length + 6, shoulder_y + dy * length)
    # Hem edge runs perpendicular to the arm.
    nx, ny = dy, -dx
    low_end = (top_end[0] - nx * end_w, top_end[1] - ny * end_w)
    armpit = (half, ARMPIT_Y)
    style = f.get("sleeve_style", "straight")
    if style != "straight" and f.get("cuff"):
        # Bishop/puff: the sleeve billows out past the arm, then gathers into a narrow cuff.
        mid = ((top_end[0] + low_end[0]) / 2, (top_end[1] + low_end[1]) / 2)
        # A short puff stays a rounded cap over the band; a long sleeve billows wider.
        wide, narrow, band = (end_w * 0.72, end_w * 0.58, 8.0) if length < 50 else (end_w * 1.2, end_w * 0.36, 11.0)
        bulge = [(mid[0] + nx * wide - dx * 14, mid[1] + ny * wide - dy * 14),
                 (mid[0] - nx * wide - dx * 14, mid[1] - ny * wide - dy * 14)]
        a, b = (mid[0] + nx * narrow, mid[1] + ny * narrow), (mid[0] - nx * narrow, mid[1] - ny * narrow)
        crown = [(shoulder_x, shoulder_y)]
        if style == "puff":  # gathered cap stands up above the shoulder line
            crown = [(shoulder_x - 3, shoulder_y + 1), (shoulder_x + 5, shoulder_y - 6), (shoulder_x + 12, shoulder_y - 2)]
        outline = crown + [bulge[0], a, b, bulge[1], armpit]
        cuff = [a, b, (b[0] + dx * band, b[1] + dy * band), (a[0] + dx * band, a[1] + dy * band)]
        return outline, cuff
    outline = [(shoulder_x, shoulder_y), top_end, low_end, armpit]
    cuff: list[tuple[float, float]] = []
    if f.get("cuff"):
        back = 9.0
        cuff = [top_end, low_end, (low_end[0] - dx * back, low_end[1] - dy * back),
                (top_end[0] - dx * back, top_end[1] - dy * back)]
    return outline, cuff


def _top(f: dict[str, Any], kind: str, back: bool) -> list[str]:
    """SVG elements for a top (shirt, tshirt, blouse, tunic)."""
    out: list[str] = []
    fit = f.get("fit", "regular")
    half = BODY_HALF.get(fit, 50.0)
    hem_y = TOP_HEM_Y.get(f.get("length", "hip"), 178.0)
    hem_half = half + (16.0 if kind == "tunic" else 4.0 if fit != "boxy" else 0.0)
    waist_half = half - (6.0 if fit == "fitted" else 1.0 if fit == "regular" else 0.0)
    dropped = f.get("shoulder") == "dropped"
    neck_half, shaped = _neck_shape(f, back)
    sleeveless = f.get("sleeve_length") == "sleeveless"
    shoulder_x = CX + half - 6 + (12 if dropped else 0) - (5 if sleeveless else 0)
    shoulder_y = SHOULDER_Y + (8 if dropped else 0)
    neck_y = SHOULDER_Y - 10
    sleeve, cuff = _sleeve(f, CX + half, shoulder_x, shoulder_y)
    # Contrast collar/cuffs are drawn in the contrast class (light on dark fabric, dark on light).
    collar_c = " contrast" if f.get("contrast_collar") else ""
    cuff_c = " contrast" if f.get("contrast_cuff") else ""
    for side in (sleeve, _mirror(sleeve)):
        if side:
            out.append(f'<polygon class="fill" points="{_pts(side)}"/>')
    for band in (cuff, _mirror(cuff)):
        if band:
            out.append(f'<polygon class="fill trim{cuff_c}" points="{_pts(band)}"/>')
            if f.get("sleeve_style", "straight") != "straight":
                # Gather marks run from the cuff seam back up the sleeve.
                (ax, ay), (bx, by), _, (dx_, dy_) = band
                ux, uy = (dx_ - ax) / 11.0, (dy_ - ay) / 11.0
                for t in (0.2, 0.4, 0.6, 0.8):
                    px, py = ax + (bx - ax) * t, ay + (by - ay) * t
                    out.append(f'<line class="detail" x1="{px:.1f}" y1="{py:.1f}" x2="{px - ux * 9:.1f}" y2="{py - uy * 9:.1f}"/>')
            if f.get("elastic_cuff"):
                (ax, ay), (bx, by), (cx_, cy_), (dx_, dy_) = band
                out.append(f'<line class="detail" x1="{(ax + dx_) / 2:.1f}" y1="{(ay + dy_) / 2:.1f}" '
                           f'x2="{(bx + cx_) / 2:.1f}" y2="{(by + cy_) / 2:.1f}" stroke-dasharray="1.5 1.5"/>')
    right = [(CX + neck_half, neck_y), (shoulder_x, shoulder_y), (CX + half, ARMPIT_Y),
             (CX + waist_half, (ARMPIT_Y + hem_y) / 2 - 8), (CX + hem_half, hem_y)]
    body = right + [(2 * CX - x, y) for x, y in reversed(right)]
    out.append(f'<polygon class="fill" points="{_pts(body)}"/>')
    if not sleeve:
        for side in (1, -1):
            line = _armhole_line(shoulder_x, shoulder_y, half)
            out.append(line if side == 1 else f'<g transform="translate({2 * CX:.0f},0) scale(-1,1)">{line}</g>')
    collar = f.get("collar", "none")
    depth = shaped or (6.0 if back else (14.0 if collar != "none" else 20.0))
    out.append(f'<path class="neck" d="{_neck(f, back, neck_half, neck_y, depth)}"/>')
    if collar == "shirt_collar":
        if back:
            out.append(f'<path class="fill trim{collar_c}" d="M{CX - neck_half:.1f},{neck_y:.1f} Q{CX:.1f},{neck_y + 9:.1f} {CX + neck_half:.1f},{neck_y:.1f} '
                       f'L{CX + neck_half + 3:.1f},{neck_y + 8:.1f} Q{CX:.1f},{neck_y + 18:.1f} {CX - neck_half - 3:.1f},{neck_y + 8:.1f} Z"/>')
        else:
            wing = [(CX + neck_half, neck_y), (CX + neck_half + 6, neck_y + 16), (CX + 2, neck_y + 26), (CX, neck_y + depth)]
            for w in (wing, _mirror(wing)):
                out.append(f'<polygon class="fill trim{collar_c}" points="{_pts(w)}"/>')
    elif collar == "peter_pan":
        for sign in (1, -1):
            x0, x1 = CX + sign * neck_half, CX + sign * 1.5
            out.append(f'<path class="fill trim{collar_c}" d="M{x0:.1f},{neck_y:.1f} C{x0 + sign * 10:.1f},{neck_y + 8:.1f} '
                       f'{x1 + sign * 12:.1f},{neck_y + 28:.1f} {x1:.1f},{neck_y + (depth if not back else 10):.1f} '
                       f'Q{CX + sign * 9:.1f},{neck_y + 8:.1f} {x0:.1f},{neck_y:.1f} Z"/>')
    elif collar == "band":
        out.append(f'<path class="fill trim{collar_c}" d="M{CX - neck_half:.1f},{neck_y:.1f} L{CX - neck_half + 1:.1f},{neck_y - 8:.1f} '
                   f'Q{CX:.1f},{neck_y - 2:.1f} {CX + neck_half - 1:.1f},{neck_y - 8:.1f} L{CX + neck_half:.1f},{neck_y:.1f} '
                   f'Q{CX:.1f},{neck_y + 8:.1f} {CX - neck_half:.1f},{neck_y:.1f} Z"/>')
    if not back and f.get("opening") == "button_front":
        top = neck_y + (44 if f.get("neckline") == "v" and collar == "none" else depth) + 2
        out.append(f'<line class="detail" x1="{CX + 5:.1f}" y1="{top:.1f}" x2="{CX + 5:.1f}" y2="{hem_y:.1f}"/>')
        y = top + 8
        while y < hem_y - 6:
            out.append(f'<circle class="button" cx="{CX:.1f}" cy="{y:.1f}" r="1.9"/>')
            y += 18
    if not back:
        for i in range(int(f.get("pockets", 0) or 0)):
            sign = -1 if i == 0 else 1
            x = CX + sign * (half * 0.5) - 10
            out.append(f'<path class="detail" d="M{x:.1f},{ARMPIT_Y - 4:.1f} h20 v20 l-10,5 l-10,-5 Z"/>')
    if kind in ("blouse", "tunic") and not back:
        for sign in (1, -1):
            out.append(f'<line class="detail" x1="{CX + sign * (half - 1):.1f}" y1="{ARMPIT_Y + 14:.1f}" '
                       f'x2="{CX + sign * (half * 0.45):.1f}" y2="{ARMPIT_Y + 8:.1f}"/>')
    if kind in ("blouse", "tunic") and back:
        for sign in (1, -1):
            x = CX + sign * half * 0.5
            out.append(f'<path class="detail" d="M{x - 2:.1f},{ARMPIT_Y + 10:.1f} L{x:.1f},{(ARMPIT_Y + hem_y) / 2 + 6:.1f} '
                       f'L{x + 2:.1f},{ARMPIT_Y + 10:.1f}"/>')
    if kind == "tunic":
        for sign in (1, -1):
            out.append(f'<line class="slit" x1="{CX + sign * hem_half:.1f}" y1="{hem_y:.1f}" '
                       f'x2="{CX + sign * (hem_half - 3):.1f}" y2="{hem_y - 16:.1f}"/>')
    out.append(f'<line class="hemline" x1="{CX - hem_half + 2:.1f}" y1="{hem_y - 5:.1f}" x2="{CX + hem_half - 2:.1f}" y2="{hem_y - 5:.1f}"/>')
    return out


def _skirt(f: dict[str, Any], back: bool) -> list[str]:
    """SVG elements for a skirt."""
    out: list[str] = []
    waist_y, hip_y = 40.0, 92.0
    hem_half, hem_y = hem_size(f, SKIRT_HEM_Y, 195.0)
    hip_half = min(54.0, hem_half)
    right = [(CX + 40, waist_y), (CX + hip_half, hip_y), (CX + hem_half, hem_y)]
    left = [(2 * CX - x, y) for x, y in reversed(right)]
    if f.get("hem") == "scalloped":
        count = max(5, round(hem_half * 2 / 18))
        step = hem_half * 2 / count
        hem = f"L{CX + hem_half:.1f},{hem_y:.1f} " + " ".join(
            f"Q{CX + hem_half - step * (i + 0.5):.1f},{hem_y + 9:.1f} {CX + hem_half - step * (i + 1):.1f},{hem_y:.1f}"
            for i in range(count))
        d = f"M{right[0][0]:.1f},{right[0][1]:.1f} L{right[1][0]:.1f},{right[1][1]:.1f} {hem} " \
            f"L{left[1][0]:.1f},{left[1][1]:.1f} L{left[2][0]:.1f},{left[2][1]:.1f} Z"
        out.append(f'<path class="fill" d="{d}"/>')
    else:
        out.append(f'<polygon class="fill" points="{_pts(right + left)}"/>')
    if f.get("waistband"):
        out.append(f'<polygon class="fill trim" points="{_pts([(CX - 40, waist_y - 9), (CX + 40, waist_y - 9), (CX + 40, waist_y), (CX - 40, waist_y)])}"/>')
    for sign in (1, -1):
        out.append(f'<line class="detail" x1="{CX + sign * 20:.1f}" y1="{waist_y:.1f}" x2="{CX + sign * 21:.1f}" y2="{waist_y + 26:.1f}"/>')
    if back:
        out.append(f'<line class="detail" x1="{CX:.1f}" y1="{waist_y:.1f}" x2="{CX:.1f}" y2="{hip_y:.1f}"/>')
    out.append(f'<line class="hemline" x1="{CX - hem_half + 3:.1f}" y1="{hem_y - 5:.1f}" x2="{CX + hem_half - 3:.1f}" y2="{hem_y - 5:.1f}"/>')
    return out


def _hem_path(right: list[tuple[float, float]], hem_half: float, hem_y: float, scallop: bool) -> str:
    """Closed path from the right side outline, across the hem, up the mirrored left side."""
    left = [(2 * CX - x, y) for x, y in reversed(right)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in right)
    if scallop:
        count = max(5, round(hem_half * 2 / 18))
        step = hem_half * 2 / count
        d += " " + " ".join(f"Q{CX + hem_half - step * (i + 0.5):.1f},{hem_y + 9:.1f} {CX + hem_half - step * (i + 1):.1f},{hem_y:.1f}"
                            for i in range(count))
    return d + " L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in left) + " Z"


def _dress(f: dict[str, Any], back: bool) -> list[str]:
    """SVG elements for a one-piece dress drawn as a single outline.

    With a waist seam the bodice is shaped in to the waist and a seam line
    is drawn; a shift dress runs straight from bust to hip with no seam.
    """
    out: list[str] = []
    seam = bool(f.get("waist_seam"))
    half, neck_y = 44.0, SHOULDER_Y - 10
    neck_half, shaped = _neck_shape(f, back)
    hem_half, hem_y = hem_size(f, DRESS_HEM_Y, 252.0)
    shoulder_x = CX + half - 6 - (5 if f.get("sleeve_length") == "sleeveless" else 0)
    sleeve, _ = _sleeve(f, CX + half, shoulder_x, SHOULDER_Y)
    for side in (sleeve, _mirror(sleeve)):
        if side:
            out.append(f'<polygon class="fill" points="{_pts(side)}"/>')
    waist_half = 38.0 if seam else 45.0
    right = [(CX + neck_half, neck_y), (shoulder_x, SHOULDER_Y), (CX + half, ARMPIT_Y),
             (CX + waist_half, DRESS_WAIST_Y), (CX + min(54.0, hem_half), DRESS_WAIST_Y + 52), (CX + hem_half, hem_y)]
    out.append(f'<path class="fill" d="{_hem_path(right, hem_half, hem_y, f.get("hem") == "scalloped")}"/>')
    if not sleeve:
        for side in (1, -1):
            line = _armhole_line(shoulder_x, SHOULDER_Y, half)
            out.append(line if side == 1 else f'<g transform="translate({2 * CX:.0f},0) scale(-1,1)">{line}</g>')
    depth = shaped or (6.0 if back else 20.0)
    out.append(f'<path class="neck" d="{_neck(f, back, neck_half, neck_y, depth)}"/>')
    if seam:
        out.append(f'<line class="detail" x1="{CX - waist_half:.1f}" y1="{DRESS_WAIST_Y:.1f}" x2="{CX + waist_half:.1f}" y2="{DRESS_WAIST_Y:.1f}"/>')
        for sign in (1, -1):  # bodice darts above the seam
            x = CX + sign * half * 0.5
            out.append(f'<path class="detail" d="M{x - 2:.1f},{DRESS_WAIST_Y:.1f} L{x:.1f},{ARMPIT_Y + 14:.1f} L{x + 2:.1f},{DRESS_WAIST_Y:.1f}"/>')
    out.append(f'<line class="hemline" x1="{CX - hem_half + 3:.1f}" y1="{hem_y - 5:.1f}" x2="{CX + hem_half - 3:.1f}" y2="{hem_y - 5:.1f}"/>')
    return out


def flat_svg(part: str, kind: str, features: dict[str, Any], color: str | None = None,
             title: str = "") -> str:
    """Front and back flat of one garment piece as a standalone SVG string."""
    fill = color_hex(color)
    dark = _luminance(fill) < 0.3
    contrast = color_hex("hitam") if not dark else color_hex("putih")
    line = "#d9dedb" if dark else "#1d2622"  # detail lines stay visible on dark fabric
    height = VIEW_H.get(part, 300.0)
    views = []
    for i, back in enumerate((False, True)):
        items = (_skirt(features, back) if part == "bottom" else _dress(features, back) if part == "dress"
                 else _top(features, kind, back))
        label = "BACK" if back else "FRONT"
        views.append(f'<g transform="translate({VIEW_PAD + i * VIEW_W},0)">{"".join(items)}'
                     f'<text x="{CX:.0f}" y="{height - 8:.0f}" text-anchor="middle" class="view">{label}</text></g>')
    style = (f".fill{{fill:{fill};stroke:#1d2622;stroke-width:1.4;stroke-linejoin:round}}"
             ".trim{fill-opacity:.92;filter:brightness(.94)}"
             f".trim.contrast{{fill:{contrast};fill-opacity:1;filter:none}}"
             f".detail,.neck,.slit{{fill:none;stroke:{line};stroke-width:1;stroke-linecap:round}}"
             f".hemline{{stroke:{line};stroke-width:.7;stroke-dasharray:3 2;opacity:.6}}"
             ".button{fill:#fff;stroke:#1d2622;stroke-width:.8}"
             ".view{font:700 10px Archivo,Arial,sans-serif;letter-spacing:.12em;fill:#5b6862}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2 * VIEW_W + 2 * VIEW_PAD:.0f} {height:.0f}" role="img">'
            f"<title>{escape(title or 'Sketsa flat')}</title><style>{style}</style>{''.join(views)}</svg>")


def plan_flat(plan: StylePlan, color: str | None = None, title: str = "") -> str:
    """Flat of what the plan will draft (falls back to raw features if undraftable)."""
    kind = plan.garment or plan.kind
    return flat_svg(plan.part, kind, view_features(plan), color, title)
