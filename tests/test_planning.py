import pytest
import numpy as np
from team_planning_for_geeks import Planner, ValidateBounds


def test_validate_bounds():
    values = np.array([0., 1., 2.])
    ValidateBounds(0., 2.)(values)

    with pytest.raises(ValueError):
        ValidateBounds(1., -1)

    with pytest.raises(ValueError):
        ValidateBounds(0., 1.)(values)

    with pytest.raises(ValueError):
        ValidateBounds(1., 2.)(values)


def test_initialize():
    planner = Planner(['harry', 'ron', 'hermione'], ['potions', 'herbology'], range(12))
    planner.initialize_values(0.)
    np.testing.assert_array_equal(planner.values, np.zeros((3,2,12)))

    with pytest.raises(ValueError):
        planner.initialize_values(1.5)
        planner.initialize_values(-1)