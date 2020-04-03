import attr

import numpy as np
import xarray as xr

@attr.s
class Planner:
    names=attr.ib()
    tasks=attr.ib()
    time=attr.ib()

    def initialize_values(self, value=0):
        full_values = np.full((len(self.names), len(self.tasks), len(self.time)), value)
        self._data = self.set_data(full_values)
        self.values = self._get_values()
    
    def set_data(self, values):
        return xr.DataArray(
            values,
            dims=('name', 'task', 'time'),
            coords=dict(name=self.names, task=self.tasks, time=self.time)
        )

    def _get_values(self):
        return self._data.values
