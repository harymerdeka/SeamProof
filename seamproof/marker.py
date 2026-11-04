"""Shape-based marker nesting for cutting, in millimetres.

Pieces are packed by their real cut outlines (seam allowance included), not
by bounding boxes. Marker coordinates: x runs across the fabric width
(selvedge to selvedge), y runs along the fabric length from the marker start.
Assumptions, all provisional workshop practice:
- Every piece is turned so its grainline runs along the fabric length; only
  0/180 degree turns are tried after that (no grain tilt).
- Single-ply, face-up spreading (gelar terbuka): cut-on-fold pieces are
  opened to full width, and the second of every non-fold pair is mirrored so
  the marker yields a left and a right piece.
- At least 5 mm between cut lines and from both selvedges and the marker
  start; the marker end gets 5 mm too.
- Nesting is greedy (longest piece first, lowest then leftmost position) on a
  2.5 mm occupancy grid, not optimal; utilisation is reported as measured
  (real piece area over fabric area), never targeted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points, polylabel, unary_union

from seamproof.allowance import ensure_allowance
from seamproof.export import _notch_point, edge_by_id
from seamproof.pattern import Panel, Pattern, panel_polygon

SPACING_MM = 5.0
GRID_MM = 2.5

XY = tuple[float, float]


@dataclass
class Placement:
    """One placed cut piece; geometry is in marker coordinates (mm)."""

    panel: str
    x: float  # bounding box of the placed cut outline
    y: float
    width: float
    height: float
    rotation_deg: int = 0
    mirrored: bool = False
    copy: int = 1
    copies: int = 1
    opened_fold: bool = False
    cut: tuple[XY, ...] = ()
    stitch: tuple[tuple[XY, ...], ...] = ()
    fold: tuple[XY, ...] = ()
    grain: tuple[XY, XY] = ((0.0, 0.0), (0.0, 0.0))
    notches: tuple[tuple[XY, XY], ...] = ()  # stitch point -> cut edge
    label_at: XY = (0.0, 0.0)
    area_mm2: float = 0.0


@dataclass
class Marker:
    """Packing result for one garment: placements plus measured usage."""

    fabric_width_mm: float
    placements: list[Placement] = field(default_factory=list)
    used_length_mm: float = 0.0
    piece_area_mm2: float = 0.0

    @property
    def utilisation(self) -> float:
        """Total piece area over consumed fabric area, 0 when empty."""
        total = self.fabric_width_mm * self.used_length_mm
        return self.piece_area_mm2 / total if total > 0 else 0.0


@dataclass
class _Shape:
    """One piece in local coordinates before placement."""

    panel: str
    cut: Polygon
    stitch: list[LineString]
    fold: LineString | None
    grain: LineString
    notches: list[LineString]


def _reflect(geom, a: XY, b: XY):
    """Mirror a geometry across the line through a and b."""
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    moved = affinity.rotate(affinity.translate(geom, -a[0], -a[1]), -angle, origin=(0, 0))
    flipped = affinity.scale(moved, 1, -1, origin=(0, 0))
    return affinity.translate(affinity.rotate(flipped, angle, origin=(0, 0)), a[0], a[1])


def _shape(panel: Panel) -> _Shape:
    """Build the full cut piece; fold pieces are opened across their fold edge."""
    cut = Polygon([(p.x, p.y) for p in panel.cut_line.points]) if panel.cut_line else panel_polygon(panel)
    stitch = [LineString([(p.x, p.y) for p in e.curve.points]) for e in panel.edges if not e.is_fold]
    grain = LineString([(p.x, p.y) for p in panel.grainline])
    notches = []
    for notch in panel.notches:
        edge = edge_by_id(panel, notch.edge_id)
        spot = None if edge is None else _notch_point(edge.curve, notch.distance_from_start_mm)
        if spot is not None:
            _, tip = nearest_points(shapely.Point(spot.x, spot.y), cut.exterior)
            notches.append(LineString([(spot.x, spot.y), (tip.x, tip.y)]))
    fold = None
    folds = [e for e in panel.edges if e.is_fold]
    if panel.cut_on_fold and folds:
        # Assumption (true for every draft so far): the fold edge is straight.
        start, end = folds[0].curve.points[0], folds[0].curve.points[-1]
        a, b = (start.x, start.y), (end.x, end.y)
        mirror = _reflect(cut, a, b)
        # Both halves share the fold line; a tiny close merges the seam exactly.
        cut = unary_union([cut, mirror]).buffer(0.01, join_style="mitre").buffer(-0.01, join_style="mitre")
        if cut.geom_type != "Polygon":
            raise ValueError(f"Panel {panel.name}: the fold cannot be opened into one piece.")
        stitch += [_reflect(line, a, b) for line in stitch]
        notches += [_reflect(line, a, b) for line in notches]
        fold = LineString([a, b])
    return _Shape(panel.name, cut, stitch, fold, grain, notches)


def _oriented(shape: _Shape, turn_deg: int, mirrored: bool) -> tuple[_Shape, float]:
    """Turn the grainline onto the fabric length, then apply 0/180 and mirror.

    Returns the shape normalised so its cut bounding box starts at (0, 0),
    plus the total rotation applied in degrees.
    """
    (x0, y0), (x1, y1) = shape.grain.coords[0], shape.grain.coords[-1]
    grain_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
    rotation = (90.0 - grain_deg + turn_deg) % 360.0
    centre = shape.cut.centroid

    def move(geom):
        geom = affinity.rotate(geom, rotation, origin=centre)
        return affinity.scale(geom, -1, 1, origin=centre) if mirrored else geom

    cut = move(shape.cut)
    min_x, min_y, _, _ = cut.bounds

    def place(geom):
        return affinity.translate(move(geom), -min_x, -min_y)

    return _Shape(shape.panel, place(shape.cut), [place(g) for g in shape.stitch],
                  place(shape.fold) if shape.fold is not None else None, place(shape.grain),
                  [place(g) for g in shape.notches]), rotation


def _mask(cut: Polygon, pad_cells: int) -> np.ndarray:
    """Conservative occupancy of the cut outline grown by half the spacing.

    A cell is marked when its centre lies within spacing/2 + half a cell
    diagonal of the piece, so every cell the grown piece touches is marked.
    Two pieces with disjoint masks are therefore at least SPACING_MM apart.
    """
    grown = cut.buffer(SPACING_MM / 2 + GRID_MM * math.sqrt(0.5))
    _, _, max_x, max_y = cut.bounds
    cols = math.ceil(max_x / GRID_MM) + 2 * pad_cells + 1
    rows = math.ceil(max_y / GRID_MM) + 2 * pad_cells + 1
    xs = (np.arange(cols) + 0.5 - pad_cells) * GRID_MM
    ys = (np.arange(rows) + 0.5 - pad_cells) * GRID_MM
    gx, gy = np.meshgrid(xs, ys)
    return shapely.contains_xy(grown, gx, gy)


def _placement_spot(grid: np.ndarray, grid_fft: np.ndarray, mask: np.ndarray, col_lo: int,
                    col_hi: int, row_lo: int) -> tuple[int, int] | None:
    """Lowest, then leftmost, mask offset that hits no occupied cell."""
    rows, cols = grid.shape
    h, w = mask.shape
    if col_hi < col_lo or rows < h:
        return None
    # Cross-correlation by FFT counts overlapping cells for every offset at once.
    padded = np.zeros(grid.shape, dtype=float)
    padded[:h, :w] = mask
    overlap = np.fft.irfft2(grid_fft * np.conj(np.fft.rfft2(padded)), s=grid.shape)
    free = overlap[row_lo:rows - h + 1, col_lo:col_hi + 1] < 0.5
    hits = np.argwhere(free)
    if hits.size == 0:
        return None
    row, col = hits[0]  # argwhere is row-major: lowest row first, then leftmost
    return int(row) + row_lo, int(col) + col_lo


def _fft_size(n: int) -> int:
    """Smallest 2^a * 3^b * 5^c >= n, a fast FFT length."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 <= best:
        p35 = p5
        while p35 <= best:
            p2 = p35
            while p2 < n:
                p2 *= 2
            best = min(best, p2)
            p35 *= 3
        p5 *= 5
    return best


def pack(pattern: Pattern, fabric_width_mm: float) -> Marker:
    """Nest every cut piece on the fabric; return placements and usage."""
    if not fabric_width_mm > 0:
        raise ValueError("Fabric width must be positive.")
    pattern = ensure_allowance(pattern)
    pad = math.ceil((SPACING_MM / 2 + GRID_MM) / GRID_MM)
    margin_cells = math.ceil(SPACING_MM / GRID_MM)
    jobs = []
    for panel in pattern.panels:
        shape = _shape(panel)
        mirror_pairs = not panel.cut_on_fold
        options = []
        for copy_no in range(1, panel.cut_count + 1):
            mirrored = mirror_pairs and copy_no % 2 == 0
            variants = []
            for turn in (0, 180):
                local, rotation = _oriented(shape, turn, mirrored)
                variants.append((local, round(rotation) % 360, _mask(local.cut, pad)))
            options.append((copy_no, mirrored, variants))
        width = options[0][2][0][0].cut.bounds[2]
        if width + 2 * SPACING_MM > fabric_width_mm:
            raise ValueError(
                f"Panel {panel.name} ({width:.0f} mm) does not fit the {fabric_width_mm:.0f} mm width.")
        for copy_no, mirrored, variants in options:
            jobs.append((panel, copy_no, mirrored, variants, variants[0][0].cut.bounds[3]))
    # Longest along the fabric first: it packed shorter markers than area-first
    # on every garment tried (e.g. shirt L 1107 vs 1262 mm at 1500 mm width).
    jobs.sort(key=lambda job: (-job[4], job[0].name, job[1]))

    cols = _fft_size(math.ceil(fabric_width_mm / GRID_MM) + 2 * pad)
    grid = np.zeros((0, cols), dtype=bool)
    marker = Marker(fabric_width_mm)
    for panel, copy_no, mirrored, variants, _length in jobs:
        best = None
        used_rows = math.ceil(marker.used_length_mm / GRID_MM) + 2 * pad + margin_cells
        needed = used_rows + max(v[2].shape[0] for v in variants) + 1
        if grid.shape[0] < needed:
            grid = np.vstack([grid, np.zeros((_fft_size(needed) - grid.shape[0], cols), dtype=bool)])
        grid_fft = np.fft.rfft2(grid.astype(float))
        for local, rotation, mask in variants:
            h, w = mask.shape
            # Grid column c spans [(c - pad) * GRID, ...); the cut bbox starts at the
            # mask origin + pad cells, so its left edge sits at col * GRID mm.
            piece_w = local.cut.bounds[2]
            col_lo = margin_cells
            col_hi = min(math.floor((fabric_width_mm - SPACING_MM - piece_w) / GRID_MM), cols - w)
            spot = _placement_spot(grid, grid_fft, mask, col_lo, col_hi, margin_cells)
            if spot is None:
                continue
            row, col = spot
            key = (row + h, row, col)
            if best is None or key < best[0]:
                best = (key, local, rotation, mask, row, col)
        if best is None:  # Cannot happen after the width check, except by bug.
            raise ValueError(f"Panel {panel.name} could not be placed — check the fabric width.")
        _key, local, rotation, mask, row, col = best
        h, w = mask.shape
        grid[row:row + h, col:col + w] |= mask
        marker.placements.append(_place(local, panel, copy_no, mirrored, rotation,
                                        col * GRID_MM, row * GRID_MM))
        last = marker.placements[-1]
        marker.used_length_mm = max(marker.used_length_mm, last.y + last.height + SPACING_MM)
        marker.piece_area_mm2 += last.area_mm2
    return marker


def _coords(geom, dx: float, dy: float) -> tuple[XY, ...]:
    """Round shifted coordinates to 0.01 mm for stable output."""
    return tuple((round(x + dx, 2), round(y + dy, 2)) for x, y in geom.coords)


def _place(local: _Shape, panel: Panel, copy_no: int, mirrored: bool, rotation: int,
           dx: float, dy: float) -> Placement:
    """Freeze one oriented shape at its marker position."""
    _, _, max_x, max_y = local.cut.bounds
    label = polylabel(local.cut, tolerance=2.0)
    grain = _coords(local.grain, dx, dy)
    return Placement(
        panel=panel.name, x=dx, y=dy, width=max_x, height=max_y, rotation_deg=rotation,
        mirrored=mirrored, copy=copy_no, copies=panel.cut_count, opened_fold=local.fold is not None,
        cut=_coords(local.cut.exterior, dx, dy),
        stitch=tuple(_coords(line, dx, dy) for line in local.stitch),
        fold=_coords(local.fold, dx, dy) if local.fold is not None else (),
        grain=(grain[0], grain[-1]),
        notches=tuple(_coords(line, dx, dy) for line in local.notches),
        label_at=(round(label.x + dx, 2), round(label.y + dy, 2)),
        area_mm2=local.cut.area)
