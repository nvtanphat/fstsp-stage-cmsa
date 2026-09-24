from __future__ import annotations


class AgeManager:
    """CMSA component-age state following Algorithm 1.

    Components with age == -1 are disabled (not included in the restricted MIP).
    Components with age >= 0 are active.
    Useful components (present in constructed or MIP solutions) are reset to 0.
    At each Adapt step (Algorithm 1 Lines 11-18), active components have their
    age incremented by 1. Components reaching age_limit are disabled (age set to -1).
    """

    def __init__(self, age_limit: int = 2) -> None:
        if age_limit < 1:
            raise ValueError("age_limit must be >= 1")
        self.age_limit = age_limit
        self.age: dict[tuple, int] = {}

    def mark_useful(self, components: set[tuple]) -> None:
        """Reset age of useful components to 0 (Algorithm 1 Lines 7 and 10)."""
        for c in components:
            self.age[c] = 0

    def adapt(self) -> None:
        """Increment age of all active components; disable those reaching age_limit (Algorithm 1 Lines 11-18)."""
        for c in list(self.age):
            if self.age[c] >= 0:
                self.age[c] += 1
                if self.age[c] >= self.age_limit:
                    self.age[c] = -1

    def active(self) -> set[tuple]:
        """Return the set of active components (age >= 0)."""
        return {c for c, age in self.age.items() if age >= 0}

    def reset(self) -> None:
        """Reset all component ages to disabled (-1)."""
        for c in self.age:
            self.age[c] = -1
