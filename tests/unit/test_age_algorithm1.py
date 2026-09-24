from __future__ import annotations

import pytest

from fstsp.algorithms.cmsa.age import AgeManager


def test_age_manager_algorithm1_flow():
    """Verify AgeManager marks useful, increments on adapt, and evicts at age_limit."""
    am = AgeManager(age_limit=2)

    comp_a = ("truck", 0, 1)
    comp_b = ("drone", 1, 2, 3)
    comp_c = ("truck", 1, 4)

    # 1. Add components -> marked useful with age 0
    am.mark_useful({comp_a, comp_b})
    assert am.active() == {comp_a, comp_b}
    assert am.age[comp_a] == 0
    assert am.age[comp_b] == 0

    # 2. Adapt iteration 1: ages increment to 1, still < age_limit
    am.adapt()
    assert am.age[comp_a] == 1
    assert am.age[comp_b] == 1
    assert am.active() == {comp_a, comp_b}

    # 3. comp_a is used again in solution, comp_c newly added; comp_b not used
    am.mark_useful({comp_a, comp_c})
    assert am.age[comp_a] == 0
    assert am.age[comp_b] == 1
    assert am.age[comp_c] == 0

    # 4. Adapt iteration 2: comp_b reaches age 2 (age_limit) -> disabled (-1)
    am.adapt()
    assert am.age[comp_a] == 1
    assert am.age[comp_b] == -1
    assert am.age[comp_c] == 1
    assert am.active() == {comp_a, comp_c}
    assert comp_b not in am.active()
