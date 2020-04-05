import pytest
import numpy as np
import pandas as pd

from team_planning_for_geeks import Planner
from team_planning_for_geeks.planning import ValidateBounds


def test_validate_bounds():
    values = np.array([0., 1., 2.])
    ValidateBounds(0., 2.)(values)

    with pytest.raises(ValueError):
        ValidateBounds(1., -1)

    with pytest.raises(ValueError):
        ValidateBounds(0., 1.)(values)

    with pytest.raises(ValueError):
        ValidateBounds(1., 2.)(values)


@pytest.fixture
def planner():
    return Planner(
        names=['harry', 'ron', 'hermione'], 
        tasks=['potions', 'herbology'], 
        time=range(12)
    )


def test_initialize(planner):
    planner.initialize_values(0.)
    np.testing.assert_array_equal(planner.values, np.zeros((3,2,12)))

    with pytest.raises(ValueError):
        planner.initialize_values(1.5)
        planner.initialize_values(-1)


def test_set_query(planner):
    planner.initialize_values(0.)
    planner.set_values(dict(name=['ron'], task=['potions'], time=[1]), 0.1)
    np.testing.assert_array_equal(
        planner.query(dict(name=['ron'], task=['potions'], time=[1])).values,
        np.array([[[0.1]]])
    )

    planner.set_values(dict(name=['ron'], task=['herbology']), np.ones(12))
    np.testing.assert_array_equal(
        planner.query(dict(name=['ron'], task=['herbology'])).values, 
        np.reshape(np.ones(12), (1,1,12))
    )
