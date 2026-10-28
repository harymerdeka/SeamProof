"""Women's woven blouse: bodice extended to hip length with a neck facing.

Assumptions: hem falls 20 mm below the hip line and 10 mm wider per side
for slight shaping; the facing is front-neck only and the back neckline is
left for a bias finish outside this block.
"""

from typing import Literal, Mapping

from seamproof.draft.bodice import (
    BLOUSE_HEM_EXTRA,
    BLOUSE_HIP_EXTENSION,
    _assemble,
)
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import WomenMeasurements
from seamproof.params import resolve_params
from seamproof.pattern import Pattern


def draft_blouse(
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
    hip_y = m.front_length + m.waist_to_hip
    hip_half = (m.hip + FABRIC_EASE[fabric].default_ease_mm) / 4
    pattern, _ = _assemble("blouse", m, fabric, p, hip_y + BLOUSE_HIP_EXTENSION,
                           hip_half + BLOUSE_HEM_EXTRA, 0, 0, True, back_opening)
    pattern.metadata["back_opening"] = back_opening
    pattern.metadata["assumptions"] += " Blouse hem 20 mm below hip; back neck bias-finished."
    return pattern
