"""Women's woven tunic: blouse block lengthened with flare, slits and a curved hem.

Assumptions: the hem is levelled to front waist plus tunic_length; side flare
widens each side at the hem only; the curved hem dips hem_curve at the centre;
slits open the side seam slit_height above the hem with a notch at each top.
"""

from typing import Literal, Mapping

from seamproof.draft.bodice import _assemble
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import WomenMeasurements
from seamproof.params import resolve_params
from seamproof.pattern import Pattern


def draft_tunic(
    measurements: WomenMeasurements | Mapping[str, float],
    fabric: str,
    params: Mapping[str, float] | None = None,
    back_opening: Literal["none", "centre_back_zip"] = "none",
) -> Pattern:
    """Draft front, back, sleeve and facing; no LLM, allowance or verifier is invoked."""
    m = measurements if isinstance(measurements, WomenMeasurements) else WomenMeasurements(**measurements)
    if fabric not in FABRIC_EASE:
        raise ValueError("Unknown fabric category")
    if back_opening not in ("none", "centre_back_zip"):
        raise ValueError("Back opening must be none or centre_back_zip")
    p = resolve_params(params)
    hip_half = (m.hip + FABRIC_EASE[fabric].default_ease_mm) / 4
    pattern, _ = _assemble("tunic", m, fabric, p, m.front_length + p["tunic_length"],
                           hip_half + p["side_flare"], p["slit_height"], p["hem_curve"],
                           True, back_opening)
    pattern.metadata["back_opening"] = back_opening
    pattern.metadata["assumptions"] += " Tunic hem from front waist; centre dips hem_curve."
    return pattern
