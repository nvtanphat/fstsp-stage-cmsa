from pathlib import Path
import importlib.util


MODULE = Path(__file__).resolve().parents[2] / "kaggle" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("kaggle_run_experiment", MODULE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def test_detects_restricted_family_from_parent_directory():
    assert mod._is_agatz_candidate(Path("/kaggle/input/data/restricted/maxradius/foo.txt"))
    assert mod._is_agatz_candidate(Path("/kaggle/input/data/restricted/novisit/foo.txt"))


def test_rejects_unrelated_txt_and_non_txt():
    assert not mod._is_agatz_candidate(Path("/kaggle/input/data/uniform/foo.txt"))
    assert not mod._is_agatz_candidate(Path("/kaggle/input/data/maxradius/foo.csv"))
