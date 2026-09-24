from pathlib import Path
import json
import numpy as np

from fstsp.domain.solution import FSTSPSolution


def test_json_safe_handles_numpy_arrays_sets_and_paths(tmp_path):
    out = tmp_path / "solution.json"
    sol = FSTSPSolution(
        feasible=False,
        objective=None,
        metadata={"arr": np.array([1.0, np.nan]), "set": {1, 2}, "path": Path("x/y")},
    )
    sol.to_json(out)
    payload = json.loads(out.read_text())
    assert payload["metadata"]["arr"] == [1.0, None]
    assert sorted(payload["metadata"]["set"]) == [1, 2]
    assert payload["metadata"]["path"] == "x/y"
