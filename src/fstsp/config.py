from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import sys

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ProblemConfig:
    trucks: int = 1
    drones: int = 1
    start_depot: str = "S"
    end_depot: str = "E"


@dataclass
class SolverConfig:
    backend: str = "cplex"
    version_reference: str = "22.11"
    threads: int = 8
    mip_emphasis: int | None = 5
    alternative_open_source_backend: str = "highs"


@dataclass
class ExactTimeLimitsConfig:
    table1_agatz_maxradius_seconds: float = 3600.0
    table2_agatz_novisit_seconds: float = 3600.0
    table3_cplex_exact_seconds: float = 7200.0


@dataclass
class CMSAConfig:
    age_limit: int = 2
    restricted_mip_time_limit_seconds: float = 15.0
    total_time_limit_seconds: float = 1800.0
    truck_sample_ratio: float = 0.65
    exact_tsp_threshold: int = 60


@dataclass
class Table1ExperimentConfig:
    method: str = "exact"
    time_limit_seconds: float = 3600.0


@dataclass
class Table2ExperimentConfig:
    method: str = "exact"
    time_limit_seconds: float = 3600.0


@dataclass
class Table3ExperimentConfig:
    exact_time_limit_seconds: float = 7200.0
    cmsa_time_limit_seconds: float = 1800.0
    restricted_mip_time_limit_seconds: float = 15.0
    age_limit: int = 2
    truck_sample_ratio: float = 0.65
    exact_tsp_threshold: int = 60
    instances_per_size: int = 10
    customer_sizes: list[int] = field(default_factory=lambda: [20, 30, 40, 50])


@dataclass
class Table4ExperimentConfig:
    age_limits: list[int] = field(default_factory=lambda: [2, 5])
    restricted_mip_time_limit_seconds: float = 15.0
    total_time_limit_seconds: float = 1800.0
    instances_per_size: int = 10
    customer_sizes: list[int] = field(default_factory=lambda: [20, 30, 40, 50])


@dataclass
class ExperimentsConfig:
    instances_per_size: int = 10
    customer_sizes: list[int] = field(default_factory=lambda: [20, 30, 40, 50])
    table4_age_limits: list[int] = field(default_factory=lambda: [2, 5])
    require_raw_results: bool = True
    validate_solutions: bool = True
    figure1_customer_size: int = 30
    protocol_type: str = "paper_protocol"
    table1: Table1ExperimentConfig = field(default_factory=Table1ExperimentConfig)
    table2: Table2ExperimentConfig = field(default_factory=Table2ExperimentConfig)
    table3: Table3ExperimentConfig = field(default_factory=Table3ExperimentConfig)
    table4: Table4ExperimentConfig = field(default_factory=Table4ExperimentConfig)


@dataclass
class AppConfig:
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    exact_time_limits: ExactTimeLimitsConfig = field(default_factory=ExactTimeLimitsConfig)
    cmsa: CMSAConfig = field(default_factory=CMSAConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)
    raw_dict: dict[str, Any] = field(default_factory=dict)

    def is_paper_protocol(self) -> bool:
        """Return True if this configuration strictly adheres to the official paper protocol."""
        return (
            self.experiments.protocol_type == "paper_protocol"
            and self.cmsa.total_time_limit_seconds == 1800.0
            and self.cmsa.restricted_mip_time_limit_seconds == 15.0
            and self.cmsa.age_limit in (2, 5)
        )


def _find_repo_root() -> Path:
    # Walk up from current file to find repo root with configs/ directory
    current = Path(__file__).resolve().parent
    for _ in range(4):
        if (current / "configs").is_dir():
            return current
        current = current.parent
    return Path.cwd()


def _parse_yaml_or_fallback(file_path: Path) -> dict[str, Any]:
    text = file_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        return data or {}
    # Minimal fallback parser if pyyaml is unexpectedly absent
    import json
    try:
        return json.loads(text)
    except Exception:
        raise RuntimeError("PyYAML is required to parse configuration files.")


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file path or return default paper protocol."""
    if config_path is None:
        root = _find_repo_root()
        default_file = root / "configs" / "paper_protocol.yaml"
        if default_file.exists():
            config_path = default_file
        else:
            return AppConfig()

    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    raw = _parse_yaml_or_fallback(p)

    problem_data = raw.get("problem", {})
    problem = ProblemConfig(
        trucks=problem_data.get("trucks", 1),
        drones=problem_data.get("drones", 1),
        start_depot=str(problem_data.get("start_depot", "S")),
        end_depot=str(problem_data.get("end_depot", "E")),
    )

    solver_data = raw.get("solver", {})
    solver = SolverConfig(
        backend=str(solver_data.get("backend", "cplex")),
        version_reference=str(solver_data.get("version_reference", "22.11")),
        threads=int(solver_data.get("threads", 8)),
        mip_emphasis=solver_data.get("mip_emphasis", 5),
        alternative_open_source_backend=str(solver_data.get("alternative_open_source_backend", "highs")),
    )

    exact_data = raw.get("exact_time_limits", {})
    exact_time_limits = ExactTimeLimitsConfig(
        table1_agatz_maxradius_seconds=float(exact_data.get("table1_agatz_maxradius_seconds", 3600.0)),
        table2_agatz_novisit_seconds=float(exact_data.get("table2_agatz_novisit_seconds", 3600.0)),
        table3_cplex_exact_seconds=float(exact_data.get("table3_cplex_exact_seconds", 7200.0)),
    )

    cmsa_data = raw.get("cmsa", {})
    cmsa = CMSAConfig(
        age_limit=int(cmsa_data.get("age_limit", 2)),
        restricted_mip_time_limit_seconds=float(cmsa_data.get("restricted_mip_time_limit_seconds", 15.0)),
        total_time_limit_seconds=float(cmsa_data.get("total_time_limit_seconds", 1800.0)),
        truck_sample_ratio=float(cmsa_data.get("truck_sample_ratio", 0.65)),
        exact_tsp_threshold=int(cmsa_data.get("exact_tsp_threshold", 60)),
    )

    exp_data = raw.get("experiments", {})
    t1_data = exp_data.get("table1", {})
    table1 = Table1ExperimentConfig(
        method=str(t1_data.get("method", "exact")),
        time_limit_seconds=float(t1_data.get("time_limit_seconds", exact_time_limits.table1_agatz_maxradius_seconds)),
    )

    t2_data = exp_data.get("table2", {})
    table2 = Table2ExperimentConfig(
        method=str(t2_data.get("method", "exact")),
        time_limit_seconds=float(t2_data.get("time_limit_seconds", exact_time_limits.table2_agatz_novisit_seconds)),
    )

    t3_data = exp_data.get("table3", {})
    table3 = Table3ExperimentConfig(
        exact_time_limit_seconds=float(t3_data.get("exact_time_limit_seconds", exact_time_limits.table3_cplex_exact_seconds)),
        cmsa_time_limit_seconds=float(t3_data.get("cmsa_time_limit_seconds", cmsa.total_time_limit_seconds)),
        restricted_mip_time_limit_seconds=float(t3_data.get("restricted_mip_time_limit_seconds", cmsa.restricted_mip_time_limit_seconds)),
        age_limit=int(t3_data.get("age_limit", cmsa.age_limit)),
        truck_sample_ratio=float(t3_data.get("truck_sample_ratio", cmsa.truck_sample_ratio)),
        exact_tsp_threshold=int(t3_data.get("exact_tsp_threshold", cmsa.exact_tsp_threshold)),
        instances_per_size=int(t3_data.get("instances_per_size", exp_data.get("instances_per_size", 10))),
        customer_sizes=list(t3_data.get("customer_sizes", exp_data.get("customer_sizes", [20, 30, 40, 50]))),
    )

    t4_data = exp_data.get("table4", {})
    table4 = Table4ExperimentConfig(
        age_limits=list(t4_data.get("age_limits", exp_data.get("table4_age_limits", [2, 5]))),
        restricted_mip_time_limit_seconds=float(t4_data.get("restricted_mip_time_limit_seconds", cmsa.restricted_mip_time_limit_seconds)),
        total_time_limit_seconds=float(t4_data.get("total_time_limit_seconds", cmsa.total_time_limit_seconds)),
        instances_per_size=int(t4_data.get("instances_per_size", exp_data.get("instances_per_size", 10))),
        customer_sizes=list(t4_data.get("customer_sizes", exp_data.get("customer_sizes", [20, 30, 40, 50]))),
    )

    experiments = ExperimentsConfig(
        instances_per_size=int(exp_data.get("instances_per_size", 10)),
        customer_sizes=list(exp_data.get("customer_sizes", [20, 30, 40, 50])),
        table4_age_limits=list(exp_data.get("table4_age_limits", [2, 5])),
        require_raw_results=bool(exp_data.get("require_raw_results", True)),
        validate_solutions=bool(exp_data.get("validate_solutions", True)),
        figure1_customer_size=int(exp_data.get("figure1_customer_size", 30)),
        protocol_type=str(exp_data.get("protocol_type", "paper_protocol")),
        table1=table1,
        table2=table2,
        table3=table3,
        table4=table4,
    )

    return AppConfig(
        problem=problem,
        solver=solver,
        exact_time_limits=exact_time_limits,
        cmsa=cmsa,
        experiments=experiments,
        raw_dict=raw,
    )


def load_paper_protocol() -> AppConfig:
    """Load the official Paper Protocol (configs/paper_protocol.yaml)."""
    root = _find_repo_root()
    return load_config(root / "configs" / "paper_protocol.yaml")


def load_smoke_protocol() -> AppConfig:
    """Load the Smoke Test Protocol (configs/smoke_test.yaml) for fast CI checks."""
    root = _find_repo_root()
    return load_config(root / "configs" / "smoke_test.yaml")
