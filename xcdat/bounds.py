"""Bounds module for functions related to coordinate bounds."""
import collections
from typing import Dict, Hashable, List, Optional, Tuple, get_args

import cf_xarray as cfxr  # noqa: F401
import numpy as np
import xarray as xr
from typing_extensions import Literal

from xcdat.logger import setup_custom_logger

logger = setup_custom_logger("root")

Coord = Literal["lat", "latitude", "lon", "longitude", "time"]
#: Tuple of supported coordinates in xCDAT functions and methods.
SUPPORTED_COORDS: Tuple[Coord, ...] = get_args(Coord)


@xr.register_dataset_accessor("bounds")
class DatasetBoundsAccessor:
    """A class to represent the DatasetBoundsAccessor.

    Examples
    ---------
    Import:

    >>> from xcdat import bounds

    Return dictionary of coordinate keys mapped to bounds DataArrays:

    >>> ds = xr.open_dataset("file_path")
    >>> bounds = ds.bounds.bounds

    Fill missing coordinate bounds in the Dataset:

    >>> ds = xr.open_dataset("file_path")
    >>> ds = ds.bounds.fill_missing()

    Get coordinate bounds if they exist:

    >>> ds = xr.open_dataset("file_path")
    >>>
    >>> # Throws error if bounds don't exist
    >>> lat_bounds = ds.bounds.get_bounds("lat") # or pass "latitude"
    >>> lon_bounds = ds.bounds.get_bounds("lon") # or pass "longitude"
    >>> time_bounds = ds.bounds.get_bounds("time")

    Add coordinates bounds if they don't exist:

    >>> ds = xr.open_dataset("file_path")
    >>>
    >>> # Throws error if bounds exist
    >>> ds = ds.bounds.add_bounds("lat")
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    @property
    def bounds(self) -> Dict[str, Optional[xr.DataArray]]:
        """Returns a mapping of coordinate and axis keys to their bounds.

        The dictionary provides all valid CF compliant keys for a coordinate.
        For example, latitude will includes keys for "lat", "latitude", and "Y".

        Returns
        -------
        Dict[str, Optional[xr.DataArray]]
            Dictionary mapping coordinate keys to their bounds.
        """
        ds = self._dataset

        bounds: Dict[str, Optional[xr.DataArray]] = {}
        for coord, bounds_name in ds.cf.bounds.items():
            bound = ds.get(bounds_name[0], None)
            bounds[coord] = bound

        return collections.OrderedDict(sorted(bounds.items()))

    def fill_missing(self) -> xr.Dataset:
        """Fills any missing bounds for supported coordinates in the Dataset.

        Returns
        -------
        xr.Dataset
        """
        for coord in [*self._dataset.coords]:
            if coord in SUPPORTED_COORDS:
                try:
                    self._dataset.cf.get_bounds(coord)
                except KeyError:
                    self._dataset = self.add_bounds(coord)

        return self._dataset

    def get_bounds(self, coord: Coord) -> xr.DataArray:
        """Get bounds for a coordinate.

        Parameters
        ----------
        coord : Coord
            The coordinate key.

        Returns
        -------
        xr.DataArray
            The coordinate bounds.

        Raises
        ------
        ValueError
            If an incorrect ``coord`` argument is passed.

        ValueError
            If bounds were not found. They must be added.
        """
        if coord not in SUPPORTED_COORDS:
            raise ValueError(
                "Incorrect `coord` argument. Supported coordinates include: Supported "
                f"arguments include: {', '.join(SUPPORTED_COORDS)}."
            )

        try:
            bounds = self._dataset.cf.get_bounds(coord)
        except KeyError:
            raise KeyError(f"{coord} bounds were not found, they must be added.")

        return bounds

    def add_bounds(self, coord: Coord, width: float = 0.5) -> xr.Dataset:
        """Add bounds for a coordinate using its data points.

        If bounds already exist, they must be dropped first.

        Parameters
        ----------
        coord : Coord
            The coordinate key.
        width : float, optional
            Width of the bounds relative to the position of the nearest points,
            by default 0.5.

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        ------
        ValueError
            If bounds already exist. They must be dropped first.
        """
        try:
            self._dataset.cf.get_bounds(coord)
            raise ValueError(
                f"{coord} bounds already exist. Drop them first to add new bounds."
            )
        except KeyError:
            dataset = self._add_bounds(coord, width)

        return dataset

    def _add_bounds(self, coord: Coord, width: float = 0.5) -> xr.Dataset:
        """Adds bounds for a coordinate using its data points.

        Parameters
        ----------
        coord : Coord
            The coordinate key.
        width : float, optional
            Width of the bounds relative to the position of the nearest points,
            by default 0.5.

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        ------
        ValueError
            If coords dimensions does not equal 1.
        ValueError
            If coords are length of <=1.

        Notes
        -----
        Based on [1]_ ``iris.coords._guess_bounds`` and [2]_ ``cf_xarray.accessor.add_bounds``

        References
        ----------

        .. [1] https://scitools-iris.readthedocs.io/en/stable/generated/api/iris/coords.html#iris.coords.AuxCoord.guess_bounds

        .. [2] https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.add_bounds.html#
        """
        da_coord: xr.DataArray = self._get_coord(coord)

        # Validate coordinate shape and dimensions
        if da_coord.ndim != 1:
            raise ValueError("Cannot generate bounds for multidimensional coordinates.")
        if da_coord.shape[0] <= 1:
            raise ValueError("Cannot generate bounds for a coordinate of length <= 1.")

        # Retrieve coordinate dimension to calculate the diffs between points.
        dim = da_coord.dims[0]
        diffs = da_coord.diff(dim)

        # Add beginning and end points to account for lower and upper bounds.
        diffs = np.insert(diffs, 0, diffs[0])
        diffs = np.append(diffs, diffs[-1])

        # Get lower and upper bounds by using the width relative to nearest point.
        # Transpose both bound arrays into a 2D array.
        lower_bounds = da_coord - diffs[:-1] * width
        upper_bounds = da_coord + diffs[1:] * (1 - width)
        bounds = np.array([lower_bounds, upper_bounds]).transpose()

        # Clip latitude bounds at (-90, 90)
        if (
            da_coord.name in ("lat", "latitude", "grid_latitude")
            and "degree" in da_coord.attrs["units"]
        ):
            if (da_coord >= -90).all() and (da_coord <= 90).all():
                np.clip(bounds, -90, 90, out=bounds)

        # Add coordinate bounds to the dataset
        dataset = self._dataset.copy()
        var_name = f"{coord}_bnds"
        dataset[var_name] = xr.DataArray(
            name=var_name,
            data=bounds,
            coords={coord: da_coord},
            dims=[coord, "bnds"],
            attrs={"is_generated": "True"},
        )
        dataset[da_coord.name].attrs["bounds"] = var_name

        return dataset

    def _get_coord(self, coord: Coord) -> xr.DataArray:
        """Get the matching coordinate in the dataset.

        Parameters
        ----------
        coord : Coord
            The coordinate key.

        Returns
        -------
        xr.DataArray
            Matching coordinate in the Dataset.

        Raises
        ------
        TypeError
            If no matching coordinate is found in the Dataset.
        """
        try:
            matching_coord = self._dataset.cf[coord]
        except KeyError:
            raise KeyError(f"No matching coordinates for coord: {coord}")

        return matching_coord


@xr.register_dataarray_accessor("bounds")
class DataArrayBoundsAccessor:
    """A class representing the DataArrayBoundsAccessor.

    Examples
    --------
    Import module:

    >>> from xcdat import bounds
    >>> from xcdat.dataset import open_dataset

    Copy coordinate bounds from parent Dataset to data variable:

    >>> ds = open_dataset("file_path") # Auto-generates bounds if missing
    >>> tas = ds["tas"]
    >>> tas.bounds._copy_from_dataset(ds)

    Return dictionary of axis and coordinate keys mapped to bounds DataArrays:

    >>> tas.bounds.bounds

    Return dictionary of coordinate keys mapped to bounds names:

    >>> tas.bounds.bounds_names
    """

    def __init__(self, dataarray: xr.DataArray):
        self._dataarray = dataarray

        # A Dataset container to store the bounds from the parent Dataset.
        # A Dataset is used instead of a dictionary so that it is interoperable
        # with cf_xarray and the DatasetBoundsAccessor class. This allows for
        # the dynamic generation of a bounds dict that is name-agnostic (e.g.,
        # "lat", "latitude", "Y" for latitude bounds). Refer to the ``bounds``
        # class property below for more information.
        self._bounds = xr.Dataset()

    def copy(self) -> xr.DataArray:
        """Copies the DataArray while maintaining accessor class attributes.

        This method is invoked when a copy of a variable is made through
        ``xcdat.variable.copy_variable()``.

        Returns
        -------
        xr.DataArray
            The DataArray within the accessor class.
        """
        return self._dataarray

    def copy_from_parent(self, dataset: xr.Dataset):
        """Copies coordinate bounds from the parent Dataset to the DataArray.

        In an xarray.Dataset, variables (e.g., "tas") and coordinate bounds
        (e.g., "lat_bnds") are stored in the Dataset's data variables as
        independent DataArrays that have no link between one another [3]_. As a
        result, this creates an issue when you need to reference coordinate
        bounds after extracting a variable to work on it independently.

        This function works around this issue by copying the coordinate bounds
        from the parent Dataset to the DataArray variable.

        Parameters
        ----------
        dataset : xr.Dataset
            The parent Dataset.

        Returns
        -------
        xr.DataArray
            The data variable with bounds coordinates in the list of coordinates.

        Notes
        -----

        .. [3] https://github.com/pydata/xarray/issues/1475

        """
        bounds = self._bounds.copy()

        coords = [*dataset.coords]
        boundless_coords = []
        for coord in coords:
            if coord in SUPPORTED_COORDS:
                try:
                    coord_bounds = dataset.cf.get_bounds(coord)
                    bounds[coord_bounds.name] = coord_bounds.copy()
                except KeyError:
                    boundless_coords.append(coord)

        if boundless_coords:
            raise ValueError(
                "The dataset is missing bounds for the following coords: "
                f"{', '.join(boundless_coords)}. Pass the dataset to"
                "`xcdat.dataset.open_dataset` to auto-generate missing bounds first"
            )

        self._bounds = bounds
        return self._dataarray

    @property
    def bounds(self) -> Dict[str, Optional[xr.DataArray]]:
        """Returns a mapping of coordinate keys to their bounds.

        Missing coordinates are handled by ``self.copy_from_parent()``.

        Returns
        -------
        Dict[str, Optional[xr.DataArray]]
            Dictionary mapping coordinate keys to their bounds.

        Notes
        -----
        Based on ``cf_xarray.accessor.CFDatasetAccessor.bounds``.
        """
        self._check_bounds_are_set()

        bounds = DatasetBoundsAccessor(self._bounds)
        return bounds.bounds

    @property
    def bounds_names(self) -> Dict[Hashable, List[str]]:
        """Returns a mapping of coordinate keys to the name of their bounds.

        Wrapper for ``cf_xarray.accessor.CFDatasetAccessor.bounds_names``.

        Returns
        -------
        Dict[Hashable, List[str]]
            Dictionary mapping valid keys to the variable names of their bounds.
        """
        self._check_bounds_are_set()

        return self._bounds.cf.bounds

    def get_bounds(self, coord: Coord) -> xr.DataArray:
        """Get bounds corresponding to a coordinate key.

        Wrapper for ``cf_xarray.accessor.CFDatasetAccessor.get_bounds``.

        Parameters
        ----------
        coord : Coord
            The coordinate key whose bounds are desired.

        Returns
        -------
        DataArray
            The bounds for a coordinate key.
        """
        self._check_bounds_are_set()

        bounds = DatasetBoundsAccessor(self._bounds)
        return bounds.get_bounds(coord)

    def _check_bounds_are_set(self):
        if len(self._bounds) == 0:
            raise ValueError(
                "Variable bounds are not set. Copy them from the parent Dataset using "
                "`<var>.bounds.copy_from_parent()`"
            )
