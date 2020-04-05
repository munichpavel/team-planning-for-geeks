import attr

import numpy as np
import pandas as pd
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
    """
    Team planning container for multiple tasks over time.

    Attributes
    ----------
    name: iterable
        Team member names, typically str but need not be
    task : iterable
        Task names, typically str but need not be
    time : iterable
        Time units

    """
    name=attr.ib()
    task=attr.ib()
    time=attr.ib()

    def initialize_values(self, value=0.):
        """
        Set all resource-allocation values to a single initial value

        Parameters
        ----------
        value : numeric
        """
        full_values = np.full((len(self.name), len(self.task), len(self.time)), value)
        self._data = self.set_data(full_values)
        self.values = self._get_values()
    
    def set_data(self, values):
        """
        Set all resource-allocation values by assigning the full 3d array
        of values

        Parameter
        ---------
        values : np.array
            Array of shape(len(self.name), len(self.task), len(self.time))

        """
        validator(values)
        res = xr.DataArray(
            values,
            dims=('name', 'task', 'time'),
            coords=dict(name=self.name, task=self.task, time=self.time)
        )
        return res

    def _get_values(self):
        return self._data.values

    def set_values(self, coords, value):
        self._set_values(coords, value)
        validator(self.values)

    def _set_values(self, coords, value):
        self._data.loc[coords] = value       

    def query(self, coords):
        """
        Query the planner, returning a (new) planner restricted to
        the given coordinates.

        Parameters
        ----------
        coords: dict of iterables
            Must be of form dict(name=<iter>, task=<iter>, time=<iter>)
        
        Returns
        -------
        res : Planner
            An instance of Planner with coords and values subsets of self
        """
        return self._query(coords)
        
    def _query(self, coords):
        res = Planner(
            name=coords.get('name', self.name),
            task=coords.get('task', self.task),
            time=coords.get('time', self.time)
        )
        res.initialize_values(0.)
        values = self._data.sel(coords).values
        res.set_values(coords, values)
        return res

    def project_along(self, dim, value):
        """
        Query for coordinate pair dim, value and project down along given dimension.

        Parameters
        ----------
        dim : str
            One of the 'name', 'task' or 'time'
        value : object
            Coordinate value of dimension dim

        Returns
        -------
        res : pandas.DataFrame
            DataFrame with index and column names from the remaining dimensions
        """
        print(dim)
        along_query = {dim: [value]}

        keep_dims = [d for d in self.query(along_query)._data.dims if d != dim]
        index = getattr(self, keep_dims[0])
        columns = getattr(self, keep_dims[1])
        proj_values = self.query(along_query).values.squeeze()
        return pd.DataFrame(
            proj_values, index=index, columns=columns
        )
