"""Live repair demo with the real configured backend (needs .env credentials).

Runs the draft-verify-correct loop on a size M woven blouse with an injected
sleeve-cap defect and prints the round log as a table. Uses real network
calls; tests must mock the LLM instead.
"""

import sys

sys.path.insert(0, ".")

from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.repair import INJECTS, repair_loop, rounds_table_md

GARMENT = "blouse"
SIZE = "M"
FABRIC = "woven"
INJECT = "sleeve_cap_too_high"
MAX_ROUNDS = 5


def main() -> int:
    """Run the live loop and print every round plus the final verdict."""
    print(f"Garment: {GARMENT} {SIZE} {FABRIC} | inject: {INJECT} ({INJECTS[INJECT]})")
    pattern, report, logs = repair_loop(
        WOMEN_SIZE_TABLE[SIZE], FABRIC, dict(INJECTS[INJECT]),
        max_rounds=MAX_ROUNDS, garment=GARMENT)
    print()
    print(rounds_table_md(logs), end="")
    print(report.summary_id())
    accepted = [log for log in logs if log.decision == "accepted"]
    assert all(log.locality_passed for log in accepted), "accepted change failed locality"
    print(f"Accepted changes passing locality: {len(accepted)}/{len(logs)}.")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
