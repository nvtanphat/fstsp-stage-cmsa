from __future__ import annotations


class AgeManager:
    """CMSA component-age state. -1 means disabled; >=0 means active."""

    def __init__(self, age_limit: int = 2) -> None:
        if age_limit < 1:
            raise ValueError("age_limit must be >= 1")
        self.age_limit = age_limit
        self.age: dict[tuple, int] = {}

    def mark_useful(self, components: set[tuple]) -> None:
        for c in components:
            self.age[c] = 0

    def adapt(self, protected: set[tuple] | None = None) -> None:
        """Increment age of all active components; disable those reaching age_limit."""
        protected = protected or set()
        for c in list(self.age):
            if self.age[c] < 0 or c in protected:
                continue
            self.age[c] += 1
            if self.age[c] >= self.age_limit:
                self.age[c] = -1

    def active(self) -> set[tuple]:
        return {c for c, age in self.age.items() if age >= 0}
