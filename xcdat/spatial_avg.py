"""Averager module that contains functions related to geospatial averaging."""
from functools import reduce
from typing import Dict, Hashable, List, Optional, Tuple, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from typing_extensions import Literal, TypedDict, get_args

from xcdat.dataset import get_data_var

#: Type alias for a dictionary of axes keys mapped to their bounds.
AxisWeights = Dict[Hashable, xr.DataArray]
#: Type alias of supported axes strings for spatial averaging.
SupportedAxes = Literal["lat", "lon"]
SUPPORTED_AXES: Tuple[SupportedAxes, ...] = get_args(SupportedAxes)

#: Type alias for a tuple of floats/ints for the regional selection bounds.
RegionAxisBounds = Tuple[Union[float, int], Union[float, int]]


@xr.register_dataset_accessor("spatial")
class DatasetSpatialAverageAccessor:
    """A class to represent the DatasetSpatialAverageAccessor."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    def avg(
        self,
        data_var: Optional[str] = None,
        axis: Union[List[SupportedAxes], SupportedAxes] = ["lat", "lon"],
        weights: xr.DataArray = None,
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
    ) -> xr.Dataset:
        """
        Calculate the spatial average for a rectilinear grid over a (optional)
        specified regional domain.

        Operations include:

        - If a regional boundary is specified, check to ensure it is within the
          data variable's domain boundary.
        - If axis weights are not provided, get axis weights for standard axes
          domains specified in ``axis``.
        - Adjust weights to conform to the specified regional boundary.
        - Compute spatial weighted average.

        Parameters
        ----------
        data_var: Optional[str], optional
            The key of the data variable inside the dataset to spatially
            average. If None, an inference to the desired data variable is
            attempted with the Dataset's "xcdat_infer" attr, by default None.
        axis : Union[List[SupportedAxes], SupportedAxes]
            List of axis dimensions or single axes dimension to average over.
            For example, ["lat", "lon"]  or "lat", by default ["lat", "lon"].
        weights : Optional[xr.DataArray], optional
            A DataArray containing the regional weights used for weighted
            averaging. ``weights`` must include the same spatial axis dimensions
            and have the same dimensional sizes as the data variable. If None,
            then weights are generated; by default None.
        lat_bounds : Optional[RegionAxisBounds], optional
            A tuple of floats/ints for the regional latitude lower and upper
            boundaries. This arg is used when calculating axis weights, but is
            ignored if ``weights`` are supplied. The lower bound cannot be
            larger than the upper bound, by default None.
        lon_bounds : Optional[RegionAxisBounds], optional
            A tuple of floats/ints for the regional longitude lower and upper
            boundaries. This arg is used when calculating axis weights, but is
            ignored if ``weights`` are supplied. The lower bound can be larger
            than the upper bound (e.g., across the prime meridian, dateline), by
            default None.

        Returns
        -------
        xr.Dataset
            Dataset with the spatially averaged variable.

        Raises
        ------
        KeyError
            If data variable does not exist in the Dataset.
        KeyError
            If data variable does not contain an "Y" axes (latitude dimension).
        KeyError
            If data variable does not contain an "X" axes (longitude dimension).
        ValueError
            If an incorrect axes is specified in ``axis``.

        Examples
        --------
        Import:

        >>> import xcdat

        Open a dataset and limit to a single variable:

        >>> ds = xcdat.open_dataset("path/to/file.nc", var="tas")

        Get global average time series:

        >>> ts_global = ds.spatial.avg("tas", axis=["lat", "lon"])["tas"]

        Get time series in Nino 3.4 domain:

        >>> ts_n34 = ds.spatial.avg("tas", axis=["lat", "lon"],
        >>>     lat_bounds=(-5, 5),
        >>>     lon_bounds=(-170, -120))["tas"]

        Get zonal mean time series:

        >>> ts_zonal = ds.spatial.avg("tas", axis=['lon'])["tas"]

        Using custom weights for averaging:

        >>> # The shape of the weights must align with the data var.
        >>> self.weights = xr.DataArray(
        >>>     data=np.ones((4, 4)),
        >>>     coords={"lat": self.ds.lat, "lon": self.ds.lon},
        >>>     dims=["lat", "lon"],
        >>> )
        >>>
        >>> ts_global = ds.spatial.avg("tas", axis=["lat","lon"],
        >>>     weights=weights)["tas"]
        """
        dataset = self._dataset.copy()
        da_data_var = get_data_var(dataset, data_var)
        axis = self._validate_axis(da_data_var, axis)

        if weights is None:
            if lat_bounds is not None:
                self._validate_region_bounds("lat", lat_bounds)
            if lon_bounds is not None:
                self._validate_region_bounds("lon", lon_bounds)
            weights = self._get_weights(axis, lat_bounds, lon_bounds)

        self._validate_weights(da_data_var, axis, weights)
        dataset[da_data_var.name] = self._averager(da_data_var, axis, weights)
        return dataset

    def _validate_axis(
        self, data_var: xr.DataArray, axis: Union[List[SupportedAxes], SupportedAxes]
    ) -> List[SupportedAxes]:
        """Validates the ``axis`` arg based on a set of criteria.

        This method checks if ``axis`` values are supported strings and if they
        exist in ``data_var``.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.
        axis : Union[List[SupportedAxes], SupportedAxes]
            List of axis dimensions or single axes dimension to average over.
            For example, ["lat", "lon"]  or "lat".

        Returns
        -------
        List[SupportedAxes]
            List of axis dimensions or single axes dimension to average over.

        Raises
        ------
        ValueError
            If any value inside ``axis`` is not supported.
        KeyError
            If any value inside ``axis`` does not exist in the ``data_var``.
        """
        if isinstance(axis, str):
            axis = [axis]

        for axes in axis:
            if axes not in SUPPORTED_AXES:
                raise ValueError(
                    "Incorrect `axis` argument. Supported axes include: "
                    f"{', '.join(SUPPORTED_AXES)}."
                )

            # Must use a try and except because native xarray is not name
            # agnostic with key access using axes names (e.g., "lat" is not
            # linked to a "latitude" key ), and cf_xarray does not support
            # `data_var.cf.get(axes, None)`.
            try:
                data_var.cf[axes]
            except KeyError:
                raise KeyError(
                    f"The data variable '{data_var.name}' is missing a '{axes}' "
                    "dimension, which is required for spatial averaging."
                )

        return axis

    def _validate_region_bounds(self, axis: SupportedAxes, bounds: RegionAxisBounds):
        """Validates the ``bounds`` arg based on a set of criteria.

        Parameters
        ----------
        axis : SupportedAxes
            The axis related to the bounds.
        bounds : RegionAxisBounds
            The axis bounds.

        Raises
        ------
        TypeError
            If ``bounds`` is not a tuple.
        ValueError
            If the ``bounds`` has 0 elements or greater than 2 elements.
        TypeError
            If the ``bounds`` lower bound is not a float or integer.
        TypeError
            If the ``bounds`` upper bound is not a float or integer.
        ValueError
            If the ``axis`` is "lat" and the ``bounds`` lower value is larger
            than the upper value.
        """
        if not isinstance(bounds, tuple):
            raise TypeError(
                "The {axis} regional bounds is not a tuple representing the lower and "
                "upper bounds, (lower, upper)."
            )

        if len(bounds) <= 0 or len(bounds) > 2:
            raise ValueError(
                "The {axis} regional bounds is not a length of 2 (lower, upper)."
            )

        lower, upper = bounds
        if not isinstance(lower, float) and not isinstance(lower, int):
            raise TypeError(
                f"The regional {axis} lower bound is not a float or an integer."
            )

        if not isinstance(upper, float) and not isinstance(upper, int):
            raise TypeError(
                f"Theregional {axis} upper bound is not a float or an integer."
            )

        # For latitude, require that the upper bound be larger than the lower
        # bound. Note that this does not apply to longitude (since it is
        # a circular axis).
        if axis == "lat" and lower >= upper:
            raise ValueError(
                f"The regional {axis} lower bound is greater than the upper. "
                "Pass a tuple with the format (lower, upper)."
            )

    def _get_weights(
        self,
        axis: List[SupportedAxes],
        lat_bounds: Optional[RegionAxisBounds],
        lon_bounds: Optional[RegionAxisBounds],
    ) -> xr.DataArray:
        """
        Get area weights for specified axes and an optional target domain.

        This method first determines the weights for individual axes based on
        the difference between the upper and lower bound. For latitude the
        weight is determined by the difference of sine(latitude). The individual
        axis weights are then combined to form a DataArray of weights that can
        be used to perform a weighted (spatial) average.

        If ``lat_bounds`` or ``lon_bounds`` are supplied, then grid cells
        outside this selected regional domain are given zero weight. Grid cells
        that are partially in this domain are given partial weight.

        Parameters
        ----------
        axis : Union[List[SupportedAxes], SupportedAxes]
            List of axis dimensions or single axes dimension to average over.
            For example, ["lat", "lon"]  or "lat".
        lat_bounds : Optional[RegionAxisBounds]
            Tuple of latitude boundaries for regional selection.
        lon_bounds : Optional[RegionAxisBounds]
            Tuple of longitude boundaries for regional selection.

        Returns
        -------
        xr.DataArray
           A DataArray containing the region weights to use during averaging.
           ``weights`` are 1-D and correspond to the specified axes (``axis``)
           in the region.

        Notes
        -----
        This method was developed for rectilinear grids only. ``_get_weights()``
        recognizes and operate on latitude and longitude, but could be extended
        to work with other standard geophysical dimensions (e.g., time, depth,
        and pressure).
        """
        BoundsByType = TypedDict(
            "BoundsByType",
            {"domain": Optional[xr.DataArray], "region": Optional[RegionAxisBounds]},
        )
        bounds: Dict[str, BoundsByType] = {
            "lat": {
                "domain": self._dataset.bounds.get_bounds("lat"),
                "region": lat_bounds,
            },
            "lon": {
                "domain": self._dataset.bounds.get_bounds("lon"),
                "region": lon_bounds,
            },
        }
        axis_weights: AxisWeights = {}

        for dim, bounds_by_type in bounds.items():
            domain_bounds = bounds_by_type["domain"]
            region_bounds = bounds_by_type["region"]

            if domain_bounds is not None:
                dom_bounds = domain_bounds.copy()

                if region_bounds is not None:
                    reg_bounds = np.array(region_bounds).astype(float)
                    if dim == "lon":
                        # Forces longitude bounds linearity for elements where
                        # the upper value is greater than the lower value (e.g.,
                        # across prime meridian or dateline) to avoid issues
                        # in operations/calculations that operate using the lon
                        # bounds.
                        dom_bounds, reg_bounds = self._force_lon_linearity(
                            dom_bounds, reg_bounds
                        )
                        # If the region bounds are equal, it means all longitude
                        # values are specified (e.g., (360, 360) == (0, 360)).
                        # As a result, set the region lon bounds to the domain
                        # lon bounds (which represents all values).
                        reg_bounds = self._set_equal_lon_bounds_to_domain(
                            domain_bounds, reg_bounds
                        )
                    dom_bounds = self._scale_domain_to_region(dom_bounds, reg_bounds)

                if dim == "lat":
                    # The area between two lines of latitude scales as the
                    # difference of the sine of latitude bounds. This scaling
                    # is applied here.
                    dom_bounds = np.sin(np.radians(dom_bounds))

                # Perform differencing to get weights and add metadata.
                dim_wts = np.abs(dom_bounds[:, 1] - dom_bounds[:, 0])
                dim_wts.attrs = dom_bounds.attrs
                dim_wts.name = dim + "_wts"
                axis_weights[dim] = dim_wts

        weights = self._combine_weights(axis, axis_weights)
        return weights

    def _force_lon_linearity(
        self, domain_bounds: xr.DataArray, region_bounds: np.ndarray
    ) -> Tuple[xr.DataArray, np.ndarray]:
        """
        Forces linearity for longitude bounds elements where the lower bound
        is greater than the upper bound.

        Linearity is ensured so that operations performed using the longitude
        axes and longitude bounds do not produce incorrect outputs. Across the
        prime meridian or dateline are examples of where the lower bound may be
        greater than the upper bound.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The domain longitude bounds.
        region_bounds : np.ndarray
            The region longitude bounds.

        Returns
        -------
        Tuple[xr.DataArray, np.ndarray]
            Tuple consisting of the domain bounds and region bounds.
        """
        # Normalize the lon axis orientation to "360" for domain and region
        # bounds for orientation compatibility in the proceeding operations.
        d_bounds: xr.DataArray = self._swap_lon_axes(domain_bounds.copy(), "360")
        r_bounds: np.ndarray = self._swap_lon_axes(region_bounds.copy(), "360")

        # Find lon bound elements where the lower bound is greater than the
        # upper bound (e.g., across prime meridian or dateline, (359.375, 0.625))
        # and force them to be linearly continuous.
        row_indices = np.where((d_bounds[:, 1] - d_bounds[:, 0]) < 0)[0]

        if len(row_indices) > 0:
            d_bounds[row_indices, 1] = d_bounds[row_indices, 1] + 360.0

        region_indices = np.where(r_bounds < float(np.min(d_bounds)))[0]
        if len(region_indices) > 0:
            r_bounds[region_indices] = r_bounds[region_indices] + 360.0

        return d_bounds, r_bounds

    def _swap_lon_axes(
        self, lon: Union[np.ndarray, xr.DataArray], to: Literal["180", "360"]
    ) -> Union[np.ndarray, xr.DataArray]:
        """
        Swap longitude axes orientation to "180" (-180 to 180) or "360" (0 to
        360).

        Parameters
        ----------
        lon : Union[np.ndarray, xr.DataArray]
            Longitude values to convert.
        to : Literal["180", "360"]
            Axis orientation to convert to.

        Returns
        -------
        Union[np.ndarray, xr.DataArray]
            Converted longitude values.

        Notes
        -----
        This does not reorder the values in any way; it only converts the values
        in-place between longitude conventions (-180 to 180) or (0 to 360).
        """
        lon_swap = lon.copy()
        if to == "180":
            inds = np.where(lon_swap > 180)
            lon_swap[inds] = lon_swap[inds] - 360
        elif to == "360":
            inds = np.where(lon_swap < 0)
            lon_swap[inds] = lon_swap[inds] + 360

        return lon_swap

    def _set_equal_lon_bounds_to_domain(
        self, domain_bounds: xr.DataArray, region_bounds: np.ndarray
    ) -> np.ndarray:
        """
        Sets region longitude bounds that have equal lower and upper values to
        the domain longitude bounds.

        If the lower and upper values are equal, it means all longitude values
        are being selected (e.g., (360, 360) == (0, 360)), which is equivalent
        to the domain bounds.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The domain longitude bounds.
        region_bounds : np.ndarray
            The region longitude bounds.

        Returns
        -------
        np.ndarray
            The region longitude bounds using the domain longitude bounds.
        """
        r_bounds = region_bounds.copy()
        if r_bounds[1] - r_bounds[0] == 0:
            r_bounds[1] = np.max(domain_bounds)
            r_bounds[0] = np.min(domain_bounds)

        return r_bounds

    def _scale_domain_to_region(
        self, domain_bounds: xr.DataArray, region_bounds: np.ndarray
    ) -> xr.DataArray:
        """
        Scale domain bounds to conform to a regional selection in order to
        calculate spatial weights.

        Axis weights are determined by the difference between the upper
        and lower boundary. If a region is selected, the grid cell
        bounds outside the selected region are adjusted using this method
        so that the grid cell bounds match the selected region bounds. The
        effect of this adjustment is to give partial weight to grid cells
        that are partially in the selected regional domain and zero weight
        to grid cells outside the selected domain.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The domain's bounds.
        region_bounds : np.ndarray
            The region bounds that the domain bounds are scaled down to.

        Returns
        -------
        xr.DataArray
            Scaled dimension bounds based on regional selection.

        Notes
        -----
        If a lower regional selection bound exceeds the upper selection bound,
        this algorithm assumes that the axis is longitude and the user is
        specifying a region that includes the prime meridian. The lower
        selection bound should not exceed the upper bound for latitude.
        """
        d_bounds = domain_bounds.copy()
        r_bounds = region_bounds.copy()

        # Since longitude is circular, the logic depends on whether the region
        # spans across the prime meridian or not. If a region does not include
        # the prime meridian, then grid cells between the upper/lower region
        # domain values are given weight. If the prime meridian is included in
        # the domain (e.g., for a left bound of 300 degrees and a right bound
        # of 20, then the grid cells in between the region bounds (20 and 300)
        # are given zero weight (or partial weight if the grid bounds overlap
        # with the region bounds).
        if r_bounds[1] >= r_bounds[0]:
            # Case 1 (simple case): not wrapping around prime meridian.
            # Adjustments for above / right of region.
            d_bounds[d_bounds[:, 0] > r_bounds[1], 0] = r_bounds[1]
            d_bounds[d_bounds[:, 1] > r_bounds[1], 1] = r_bounds[1]
            # Adjustments for below / left of region.
            d_bounds[d_bounds[:, 0] < r_bounds[0], 0] = r_bounds[0]
            d_bounds[d_bounds[:, 1] < r_bounds[0], 1] = r_bounds[0]

        else:
            # Case 2: wrapping around prime meridian [for longitude only]
            domain_lowers = d_bounds[:, 0]
            domain_uppers = d_bounds[:, 1]
            region_lower, region_upper = r_bounds

            # Grid cell stradling lower boundary.
            inds = np.where(
                (domain_lowers < region_lower) & (domain_uppers > region_lower)
            )[0]
            d_bounds[inds, 0] = region_lower

            # Grid cells in between boundaries (i.e., outside selection domain).
            inds = np.where(
                (domain_lowers >= region_upper) & (domain_uppers <= region_lower)
            )[0]
            # Set upper and lower grid cell boundaries to upper edge of
            # regional domain. This will mean the grid cell upper and lower
            # boundary are equal. Therefore their difference will be zero
            # and their weight will also be zero.
            d_bounds[inds, :] = region_upper

            # Grid cell stradling upper boundary.
            inds = np.where(
                (domain_lowers < region_upper) & (domain_uppers > region_upper)
            )[0]
            d_bounds[inds, 1] = r_bounds[1]

        return d_bounds

    def _combine_weights(
        self, axis: List[SupportedAxes], axis_weights: AxisWeights
    ) -> xr.DataArray:
        """Generically rescales axis weights for a given region.

        This method creates an n-dimensional weighting array by performing
        matrix multiplication for a list of specified axes using a dictionary of
        axis weights.

        Parameters
        ----------
        axis : List[SupportedAxes]
            List of axes that should be weighted.
        axis_weights : AxisWeights
            Dictionary of axis weights, where key is axes and value is the
            corresponding DataArray of weights.

        Returns
        -------
        xr.DataArray
            A DataArray containing the region weights to use during averaging.
            ``weights`` are 1-D and correspond to the specified axes (``axis``)
            in the region.
        """
        weights = {
            axes: weights for axes, weights in axis_weights.items() if axes in axis
        }
        region_weights = reduce((lambda x, y: x * y), weights.values())
        return region_weights

    def _validate_weights(
        self, data_var: xr.DataArray, axis: List[SupportedAxes], weights: xr.DataArray
    ):
        """Validates the ``weights`` arg based on a set of criteria.

        This methods checks for the dimensional alignment between the
        ``weights`` and ``data_var``. It assumes that ``data_var`` has the same
        keys that are specified  in ``axis``, which has already been validated
        using ``self._validate_axis()`` in ``self.avg()``.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable used for validation with user supplied weights.
        axis : List[SupportedAxes]
            List of axes dimension(s) average over.
        weights : xr.DataArray
            A DataArray containing the region area weights for averaging.
            ``weights`` must include the same spatial axis dimensions found in
            ``axis`` and ``data_var``, and the same axis dims sizes as
            ``data_var``.
        Raises
        ------
        KeyError
            If ``weights`` does not include the latitude dimension.
        KeyError
            If ``weights`` does not include the longitude dimension.
        ValueError
            If the axis dimension sizes between ``weights`` and ``data_var``
            are misaligned.
        """
        # Check that the supplied weights include lat and lon dimensions.
        lat_key = data_var.cf.axes["Y"][0]
        lon_key = data_var.cf.axes["X"][0]

        if "lat" in axis and lat_key not in weights.dims:
            raise KeyError(f"Check weights DataArray includes {lat_key} dimension.")

        if "lon" in axis and lon_key not in weights.dims:
            raise KeyError(f"Check weights DataArray includes {lon_key} dimension.")

        # Check the weight dim sizes equal data var dim sizes.
        dim_sizes = {key: data_var.sizes[key] for key in weights.sizes.keys()}
        for dim, size in weights.sizes.items():
            if size != dim_sizes[dim]:
                raise ValueError(
                    f"The axis dimension sizes between supplied `weights` {dict(weights.sizes)} "
                    f"and the data variable {dim_sizes} are misaligned."
                )

    def _averager(
        self, data_var: xr.DataArray, axis: List[SupportedAxes], weights: xr.DataArray
    ):
        """Perform a weighted average of a data variable.

        This method assumes all specified axes in ``axis`` exists in the data
        variable. Validation for this criteria is performed in
        ``_validate_weights()``.

        Operations include:

        - Masked (missing) data receives zero weight.
        - Perform weighted average over user-specified axes/axis.

        Parameters
        ----------
        data_var : xr.DataArray
            Data variable inside a Dataset.
        axis : List[SupportedAxes]
            List of axis dimensions or single axis dimension to average over.
            For example, ["lat", "lon"]  or "lat".
        weights : xr.DataArray
            A DataArray containing the region area weights for averaging.
            ``weights`` must include the same spatial axis dimensions and have
            the same sizes as the data variable.

        Returns
        -------
        xr.DataArray
            Variable that has been reduced via a weighted average.

        Notes
        -----
        ``weights`` must be a DataArray and cannot contain missing values.
        Missing values are replaced with 0 using ``weights.fillna(0)``.
        """
        weights = weights.fillna(0)
        with xr.set_options(keep_attrs=True):
            weighted_mean = data_var.weighted(weights).mean(tuple(axis))
            return weighted_mean
