from pathlib import Path

import numpy as np
import pytest

from fstsp.data.agatz_parser import load_geometric_instance


def test_agatz_parser_restrictions(tmp_path: Path):
    p = tmp_path / "tiny.txt"
    p.write_text(
        "1 0.5 4\n0 0 depot\n2 0 a\n0 2 b\n2 2 c\n#MAXFLY 10\n#NOVISIT 2\n"
    )
    inst = load_geometric_instance(p)
    assert inst.n == 3
    assert inst.drone_speed == 2.0
    assert inst.drone_allowed.tolist() == [True, False, True]
    # #MAXFLY is a distance limit; factor 0.5 converts it to time 5.
    assert inst.drone_endurance == pytest.approx(5.0)


def test_agatz_unrestricted_uses_finite_200_percent_equivalent(tmp_path: Path):
    p = tmp_path / "unrestricted.txt"
    p.write_text("1 0.5 3\n0 0 depot\n3 0 a\n0 4 b\n")
    inst = load_geometric_instance(p)
    # max pairwise distance is 5; unrestricted equivalent is 2*5 distance,
    # multiplied by the drone factor 0.5 -> endurance time 5.
    assert np.isfinite(inst.drone_endurance)
    assert inst.drone_endurance == pytest.approx(5.0)
