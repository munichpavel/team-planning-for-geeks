import attr

import numpy as np
import xarray as xr

@attr.s
class ValidateBounds:
    """
    Validate that values satisfy instance bounds
    
    Attributes
    ----------
    lower_bound : float
    upper_bound : float
    """
    lower_bound = attr.ib()
    upper_bound = attr.ib()
    @upper_bound.validator
    def check(self, attribute, value):
        if value < self.lower_bound:
            raise ValueError('Upper bound may not be less than lower')
    
    def __call__(self, values):
        """Wrapper to validate arguments of f"""

        if not self.all_values_geq_lower_bound(values):
            raise ValueError(f"Some values below {self.lower_bound}")
        if not self.all_values_leq_upper_bound(values):
            raise ValueError(f"Some values above {self.upper_bound}")

    def all_values_geq_lower_bound(self, values):
        return np.all(values >= self.lower_bound)

    def all_values_leq_upper_bound(self, values):
        return np.all(values <= self.upper_bound)

validator = ValidateBounds(0., 1.)
@attr.s
class Planner:
    names=attr.ib()
    tasks=attr.ib()
    time=attr.ib()

    def initialize_values(self, value=0.):
        full_values = np.full((len(self.names), len(self.tasks), len(self.time)), value)
        self._data = self.set_data(full_values)
        self.values = self._get_values()
    
    def set_data(self, values):
        validator(values)
        res = xr.DataArray(
            values,
            dims=('name', 'task', 'time'),
            coords=dict(name=self.names, task=self.tasks, time=self.time)
        )
        return res

    def _get_values(self):
        return self._data.values

    
