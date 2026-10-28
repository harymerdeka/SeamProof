"""Body measurements in mm; ease belongs in drafting parameters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

SIZE_TABLE_LABEL = "starting values — replace with workshop data"


@dataclass(frozen=True)
class TShirtMeasurements:
    """Body dimensions in mm, not finished garment dimensions.

    Assumptions: chest, waist, hip and bicep are full circumferences;
    neck_width is a straight width; shoulder_width spans both
    shoulders. Lengths and armhole_depth are workshop input dimensions.
    These are starting conventions, not a validated sizing standard.
    """

    chest: float
    waist: float
    hip: float
    shoulder_width: float
    back_length: float
    sleeve_length: float
    armhole_depth: float
    neck_width: float
    bicep: float

    def __post_init__(self) -> None:
        """Require positive, finite measurements."""
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Measurements must be positive finite millimetres")

    def to_dict(self) -> dict[str, float]:
        """Return named measurements for the pattern record."""
        return asdict(self)


# starting values — replace with workshop data
# Illustrative adult body dimensions only; not an Indonesian sizing standard.
SIZE_TABLE: dict[str, TShirtMeasurements] = {
    "S": TShirtMeasurements(880, 760, 900, 400, 640, 200, 210, 150, 280),
    "M": TShirtMeasurements(960, 840, 980, 420, 660, 210, 220, 160, 300),
    "L": TShirtMeasurements(1040, 920, 1060, 440, 680, 220, 230, 170, 320),
    "XL": TShirtMeasurements(1120, 1000, 1140, 460, 700, 230, 240, 180, 340),
}

WOMEN_SIZE_TABLE_LABEL = "starting values — replace with workshop data"


@dataclass(frozen=True)
class WomenMeasurements:
    """Women's body dimensions in mm, not finished garment dimensions.

    Assumptions: bust, high_bust, waist, hip, bicep and wrist are full
    circumferences; shoulder_width spans both shoulders; shoulder_length
    is one shoulder seam (neck point to tip); front_length runs from the
    shoulder-neck point over the bust to the waist; back_length runs from
    the nape to the waist; apex_height/apex_distance locate the bust apex
    from the shoulder-neck point; waist_to_hip is vertical waist to hip.
    These are starting conventions, not a validated sizing standard.
    """

    bust: float
    high_bust: float
    waist: float
    hip: float
    shoulder_width: float
    shoulder_length: float
    front_length: float
    back_length: float
    apex_height: float
    apex_distance: float
    waist_to_hip: float
    armhole_depth: float
    sleeve_length: float
    bicep: float
    wrist: float

    def __post_init__(self) -> None:
        """Require positive finite values and a workable bust difference."""
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Measurements must be positive finite millimetres")
        if self.bust <= self.high_bust:
            raise ValueError("Bust must exceed high bust to form a bust dart")
        if self.front_length <= self.apex_height:
            raise ValueError("Front length must reach past the bust apex")

    def to_dict(self) -> dict[str, float]:
        """Return named measurements for the pattern record, in mm."""
        return asdict(self)

    @classmethod
    def from_cm(cls, values: Mapping[str, float]) -> WomenMeasurements:
        """Build from centimetre inputs, storing millimetres."""
        return cls(**{name: float(value) * 10 for name, value in dict(values).items()})


# starting values — replace with workshop data
# Illustrative adult body dimensions only; not an Indonesian sizing standard.
WOMEN_SIZE_TABLE: dict[str, WomenMeasurements] = {
    "S": WomenMeasurements(840, 770, 660, 900, 370, 120, 410, 385, 250, 180, 200, 210, 560, 275, 150),
    "M": WomenMeasurements(920, 840, 740, 980, 390, 125, 430, 400, 260, 190, 210, 220, 570, 290, 160),
    "L": WomenMeasurements(1000, 910, 820, 1060, 410, 130, 450, 415, 270, 200, 220, 230, 580, 305, 170),
    "XL": WomenMeasurements(1080, 980, 900, 1140, 430, 135, 470, 430, 280, 210, 230, 240, 590, 320, 180),

}
