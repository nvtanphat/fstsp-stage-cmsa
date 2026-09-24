import numpy as np
import pytest

from fstsp.algorithms.tsp.mtz import solve_tsp_mtz


def dist():
    return np.ones((4, 4)) - np.eye(4)


def test_tsp_rejects_nonpositive_time_limit():
    with pytest.raises(ValueError, match="time_limit"):
        solve_tsp_mtz([1, 2], dist(), 0, 3, time_limit=0.0)


def test_tsp_rejects_duplicate_customers():
    with pytest.raises(ValueError, match="duplicates"):
        solve_tsp_mtz([1, 1], dist(), 0, 3)


def test_tsp_rejects_start_or_end_in_customers():
    with pytest.raises(ValueError, match="exclude start/end"):
        solve_tsp_mtz([0, 1], dist(), 0, 3)
