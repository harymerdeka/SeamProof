"""Edge finishes shared by drafts: bias binding strips and a straight V neckline.

Assumptions to tune with samples (a pattern maker should review):
- Bias binding is a double-fold strip cut on the true bias, BIAS_CUT_WIDTH_MM
  wide for an 8 mm finish. Its attach edge equals the sewn edge it binds (no
  stretch-shortening); only the two short ends carry seam allowance, joined
  into a loop on an armhole or turned in at a front opening.
- Sleeveless armholes sit SLEEVELESS_RAISE_MM higher than the sleeved block
  and the shoulder is cut in SLEEVELESS_CUT_IN_MM at the armhole, so the
  armhole hugs the body without a sleeve.
- A V neckline is a straight line from the neck point to centre front.
"""

from __future__ import annotations

from seamproof.pattern import Curve, Edge, Notch, Panel, Point

BIAS_CUT_WIDTH_MM = 32.0
SLEEVELESS_RAISE_MM = 15.0
SLEEVELESS_CUT_IN_MM = 10.0
V_SAMPLES = 16  # points along a V neckline, so offsets and facings follow it closely


def bias_strip(name: str, length: float, partner: str, cut_count: int) -> Panel:
    """Rectangular bias strip whose attach edge is `length` mm, grainline at 45 degrees."""
    if length <= 0:
        raise ValueError(f"{name}: binding length must be positive")
    w = BIAS_CUT_WIDTH_MM
    a, b, c, d = Point(0, 0), Point(length, 0), Point(length, w), Point(0, w)
    edges = [Edge(f"{name}.attach", Curve((a, b)), partner, needs_allowance=False),
             Edge(f"{name}.end_r", Curve((b, c)), f"{name}.end_l"),
             Edge(f"{name}.outer", Curve((c, d)), needs_allowance=False),
             Edge(f"{name}.end_l", Curve((d, a)), f"{name}.end_r")]
    mid, reach = Point(length / 2, w / 2), w / 2 - 4
    grain = (Point(mid.x - reach, mid.y - reach), Point(mid.x + reach, mid.y + reach))
    return Panel(name, edges, grain, [Notch(f"{name}.attach", length / 2)], cut_count=cut_count)


def v_line(start: Point, end: Point) -> Curve:
    """Straight V neckline from start to end, sampled evenly."""
    return Curve(tuple(Point(start.x + (end.x - start.x) * i / V_SAMPLES, start.y + (end.y - start.y) * i / V_SAMPLES)
                       for i in range(V_SAMPLES + 1)))
