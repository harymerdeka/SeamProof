"""Provisional fabric ease, in mm; tune all values with real sewn samples."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FabricEase:
    """Full body circumference ease ranges and total sleeve-cap ease range."""

    chest_ease_range_mm: tuple[float, float]
    waist_ease_range_mm: tuple[float, float]
    hip_ease_range_mm: tuple[float, float]
    default_ease_mm: float
    sleeve_cap_ease_range_mm: tuple[float, float]


# Assumptions only: these are not validated fabric or workshop specifications.
# The default applies separately to chest, waist and hip circumferences.
FABRIC_EASE: dict[str, FabricEase] = {
    "knit_low_stretch": FabricEase((20, 100), (20, 100), (20, 100), 60, (0, 10)),
    "knit_high_stretch": FabricEase((-40, 60), (-40, 60), (-40, 60), 20, (-5, 5)),
    "woven": FabricEase((80, 160), (80, 160), (80, 160), 100, (5, 20)),
}
