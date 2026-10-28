"""Starting size table and mm measurement tests."""

from dataclasses import replace

import pytest

from seamproof.measurements import (
    SIZE_TABLE,
    SIZE_TABLE_LABEL,
    TShirtMeasurements,
    WOMEN_SIZE_TABLE,
    WOMEN_SIZE_TABLE_LABEL,
    WomenMeasurements,
)


def test_starting_table() -> None:
    """The table covers S/M/L/XL with nine mm measurements each."""
    assert SIZE_TABLE_LABEL == "starting values — replace with workshop data"
    assert list(SIZE_TABLE) == ["S", "M", "L", "XL"]
    assert len(SIZE_TABLE["M"].to_dict()) == 9
    for size in SIZE_TABLE.values():
        assert TShirtMeasurements(**size.to_dict()) == size
        assert size.chest > 800  # Circumference in mm, not centimetres.


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_measurements(value: float) -> None:
    """Non-positive or non-finite measurements are rejected."""
    with pytest.raises(ValueError, match="positive finite"):
        replace(SIZE_TABLE["M"], chest=value)


def test_women_starting_table() -> None:
    """The women's table covers S/M/L/XL with fifteen mm measurements each."""
    assert WOMEN_SIZE_TABLE_LABEL == "starting values — replace with workshop data"
    assert list(WOMEN_SIZE_TABLE) == ["S", "M", "L", "XL"]
    assert len(WOMEN_SIZE_TABLE["M"].to_dict()) == 15
    for size in WOMEN_SIZE_TABLE.values():
        assert WomenMeasurements(**size.to_dict()) == size
        assert size.bust > size.high_bust
        assert size.front_length > size.apex_height


def test_women_from_cm_stores_mm() -> None:
    """Centimetre inputs convert to millimetres on the way in."""
    centimetres = {name: value / 10 for name, value in WOMEN_SIZE_TABLE["M"].to_dict().items()}
    assert WomenMeasurements.from_cm(centimetres) == WOMEN_SIZE_TABLE["M"]
    assert WomenMeasurements.from_cm(centimetres).bust == 920


@pytest.mark.parametrize("change", [{"bust": 840}, {"front_length": 200}, {"wrist": 0}])
def test_invalid_women_measurements(change: dict) -> None:
    """Missing bust difference, short fronts and bad values are rejected."""
    with pytest.raises(ValueError):
        WomenMeasurements(**{**WOMEN_SIZE_TABLE["M"].to_dict(), **change})
