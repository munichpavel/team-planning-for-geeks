import numpy as np
from team_planning_for_geeks.planning import Planner

def test_init():
    planner = Planner(['harry', 'ron', 'hermione'], ['potions', 'herbology'], range(12))
    planner.initialize_values(0)
    np.testing.assert_array_equal(planner.values, np.zeros((3,2,12)))