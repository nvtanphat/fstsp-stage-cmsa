import numpy as np
import pytest

from fstsp.domain.instance import FSTSPInstance


def base(**overrides):
    kwargs = dict(
        name="x",
        coords=np.array([[1.0, 2.0]]),
        depot_coord=np.array([0.0, 0.0]),
        truck_speed=1.0,
        drone_speed=2.0,
        drone_endurance=10.0,
        launch_time=0.0,
        recovery_time=0.0,
    )
    kwargs.update(overrides)
    return FSTSPInstance(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("truck_speed", np.nan),
        ("drone_speed", np.inf),
        ("drone_endurance", np.nan),
        ("launch_time", np.inf),
        ("recovery_time", np.nan),
    ],
)
def test_rejects_nonfinite_numeric_parameters(field, value):
    with pytest.raises(ValueError, match="finite"):
        base(**{field: value})


def test_rejects_zero_customers():
    with pytest.raises(ValueError, match="at least one customer"):
        base(coords=np.empty((0, 2)))
