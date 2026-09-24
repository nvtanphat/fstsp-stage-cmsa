import json
import math

from fstsp.domain.solution import FSTSPSolution


def test_solution_json_is_strict_and_sanitizes_nonfinite(tmp_path):
    sol = FSTSPSolution(
        feasible=False,
        objective=None,
        mip_gap=math.inf,
        metadata={"nan": math.nan, "nested": [1.0, math.inf]},
    )
    path = tmp_path / "solution.json"
    sol.to_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mip_gap"] is None
    assert payload["metadata"]["nan"] is None
    assert payload["metadata"]["nested"][1] is None
