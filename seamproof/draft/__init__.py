"""Deterministic parameter-to-geometry drafting functions."""

from seamproof.draft.blouse import draft_blouse
from seamproof.draft.bodice import draft_bodice
from seamproof.draft.dress import draft_dress
from seamproof.draft.shirt import draft_shirt
from seamproof.draft.skirt import draft_skirt
from seamproof.draft.tshirt import draft_tshirt
from seamproof.draft.tunic import draft_tunic

REGISTRY = {
    "tshirt": draft_tshirt,
    "bodice": draft_bodice,
    "blouse": draft_blouse,
    "tunic": draft_tunic,
    "shirt": draft_shirt,
    "skirt": draft_skirt,
    "dress": draft_dress,
}

__all__ = ["REGISTRY", "draft_bodice", "draft_blouse", "draft_dress", "draft_shirt", "draft_skirt", "draft_tshirt", "draft_tunic"]
