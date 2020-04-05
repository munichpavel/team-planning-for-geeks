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



# Not defined as fixtures, as they are needed in paramatrization
names = ['harry', 'ron', 'hermione']
tasks = ['potions', 'herbology']
tenors = range(12)


@pytest.fixture
def planner():
    return Planner(
        name=names, 
        task=tasks, 
        time=tenors
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

@pytest.mark.parametrize(
    "init_value,project_along,expected",
    [
        (0.2, ('task', 'potions'), pd.DataFrame(
            0.2*np.ones((len(names), len(tenors))),
            index=names, columns=tenors
        )),
        (0.8, ('name', 'hermione'), pd.DataFrame(
             0.8*np.ones((len(tasks), len(tenors))),
            index=tasks, columns=tenors
        )),
        (0.3, ('time', 1), pd.DataFrame(
            0.3*np.ones((len(names), len(tasks))),
            index=names, columns=tasks
        ))
    ]
)
def test_project_along(planner, init_value, project_along, expected):
    planner.initialize_values(init_value)
    print(planner.values)
    res = planner.project_along(*project_along)
    pd.testing.assert_frame_equal(res, expected)
