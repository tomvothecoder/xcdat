"""Functions related to calculating climatology cycles and departures."""

from typing import Dict, Optional, Tuple, Union

import cf_xarray as cfxr  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Literal, get_args

from xcdat import bounds, logger  # noqa: F401
from xcdat.dataset import get_data_var

logging = logger.setup_custom_logger("root")

# FREQUENCIES
# ===========
# Type alias for all available frequencies.
Frequency = Union[Literal["day", "month", "season", "year", "month"]]
#: Tuple of available frequencies for the ``frequency`` param.
FREQUENCIES = ("hour", "day", "month", "season", "year", "month")

# DATETIME COMPONENTS
# ===================
# Type alias representing xarray DateTime components.
DateTimeComponent = Literal["hour", "day", "month", "season", "year"]

# Maps available frequencies to xarray DateTime components for xarray operations.
# This is simple groupby
# TODO: Add support for custom seasons freq
# TODO: Add support for diurnalNNN freq
FREQS_TO_DATETIME: Dict[str, Tuple[DateTimeComponent, ...]] = {
    "day": ("month", "day"),
    "season": ("season",),
    "month": ("month",),
}

# DJF CLIMATOLOGY SPECIFIC
# ========================
# Type alias for the DJF season being continuous or discontinuous.
DJFType = Literal["cont", "discont"]
# Tuple of DJF season type params.
DJF_TYPES = get_args(DJFType)


@xr.register_dataset_accessor("climo")
class DatasetClimatologyAccessor:
    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    def cycle(
        self,
        freq: Frequency,
        data_var: str = None,
        is_weighted: bool = True,
        djf_type: DJFType = "cont",
    ) -> xr.Dataset:
        """Calculates a data variable's climatology cycle.

        The original data variable is preserved, which is used for calculating
        departure.

        Parameters
        ----------
        freq : Frequency
            The frequency of time to group by. Available aliases:
            - ``"day"`` for daily cycle climatology.
            - ``"season"` for seasonal cycle climatology.
            - ``"month"`` for annual cycle climatology.
            - ``"JAN", "FEB", ..., or "DEC"`` for specific monthly climatology.
            - Averages the month across all seasons.
            - ``"DJF", "MAM", "JJA", or "SON"`` for specific season climatology.
            - Average the season across all years.

            Refer to ``FREQUENCIES`` for a complete list of available options.
        data_var: Optional[str], optional
            The key of the data variable in the dataset to calculate climatology
            on. If None, an inference to the desired data variable is attempted
            with the Dataset's "xcdat_infer" attr, by default None.
        is_weighted : bool, optional
            Perform grouping using weighted averages, by default True.
            Time bounds, leap years, and month lengths are considered.
        djf_type : DJFType, optional
            Whether the DJF season is continuous ("cont", previous year Dec) or
            discontinuous ("discont", same year Dec), by default ``"cont"``.

            - ``cont"`` for a continuous DJF (previous year Dec)
            - ``"discont"`` for a discontinuous DJF (same year Dec)

            Seasonally continuous December (``"cont"``) refers to continuity
            between December and January. DJF starts on the first year Dec and
            second year Jan/Feb, and ending on the second to last year Dec and
            last year Jan + Feb). Incomplete seasons are dropped, which includes
            the start year Jan/ Feb and end year Dec

            - Example Date Range: Jan/2015 - Dec/2017
            - Dropped incomplete seasons -> Jan/2015, Feb/2015, and Dec/2017

            - Start -> Dec/2015, Jan/2016, Feb/2016
            - End -> Dec/2016, Jan/2017, Feb/2017

            Seasonally discontinuous December (``"discont"``) refers to
            discontinuity between Feb and Dec. DJF starts on the first year
            Jan/Feb/Dec, and ending on the last year Jan/Feb/Dec. This is the
            default xarray behavior when grouping by season.

            - Example Date Range: Jan/2015 - Dec/2017

            - Start -> Jan/2015, Feb/2015, Dec/2015
            - End -> Jan/2017, Feb/2017, Dec/2017

        Returns
        -------
        xr.Dataset
            Dataset containing the climatology cycle for a variable.

        Raises
        ------
        ValueError
            If incorrect ``frequency`` argument is passed.
        KeyError
            If the dataset does not have "time" axis.

        Examples
        --------
        Import:

        >>> import xarray as xr
        >>> from xcdat.climatology import climatology, departure
        >>> ds = xr.open_dataset("file_path")

        Get daily, seasonal, or annual weighted climatology for a variable:

        >>> ds_climo_daily = ds.climo.cycle("day", data_var="ts")
        >>> ds_climo_daily.ts
        >>>
        >>> ds_climo_seasonal = ds.climo.cycle("season", data_var="ts")
        >>> ds_climo_seasonal.ts
        >>>
        >>> ds_climo_annual = ds.climo.cycle("month", data_var="ts")
        >>> ds_climo_annual.ts

        Get monthly, seasonal, or month unweighted climatology for a variable:
        >>> ds_climo_daily = ds.climo.cycle("day", data_var="ts", is_weighted=False)
        >>> ds_climo_daily.ts
        >>>
        >>> ds_climo_seasonal = ds.climo.cycle("season", data_var="ts", is_weighted=False)
        >>> ds_climo_seasonal.ts
        >>>
        >>> ds_climo_annual = ds.climo.cycle("month", data_var="ts", is_weighted=False)
        >>> ds_climo_annual.ts


        Access Dataset attribute for climatology operation info:

        >>> ds_climo_monthly.ts.operation
        {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
        >>> ds_climo_monthly.ts.attrs["operation"]
        {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
        """
        # TODO: Add support for climatology year chunking
        if self._dataset.cf.dims.get("time") is None:
            raise KeyError(
                "This dataset does not have a 'time' dimension. Cannot calculate climatology."
            )

        if freq not in FREQUENCIES:
            freqs = ", ".join(f'"{word}"' for word in FREQUENCIES)
            raise ValueError(
                f"Incorrect `frequency` argument. Supported frequencies include: {freqs}."
            )

        if djf_type not in DJF_TYPES:
            djf_types = ", ".join(f'"{word}"' for word in DJF_TYPES)
            raise ValueError(
                f"Incorrect `djf_type` argument. Supported DJF types include: {djf_types}"
            )

        ds_climo = self._dataset.copy()
        da_data_var = get_data_var(ds_climo, data_var)
        ds_climo[data_var] = self._group_data(
            da_data_var.copy(), "climatology", freq, is_weighted, djf_type
        )
        # Preserve the original variable so that is can be used for calculating
        # departure.
        ds_climo[f"{data_var}_original"] = da_data_var
        return ds_climo

    def departure(self, data_var: str = None) -> xr.Dataset:
        """Calculates departures (anomalies) for a data variable's climatology.

        In climatology, “anomalies” refer to the difference between observations and
        typical weather for a particular season.

        The data variable from the dataset is grouped used the same frequency and
        weights as the data variable climatology. Afterwards, the departure is
        derived using the formula: variable - variable climatology.

        Parameters
        ----------
        data_var: Optional[str], optional
            The key of the data variable in the dataset to calculate climatology
            on. If None, an inference to the desired data variable is attempted
            with the Dataset's "xcdat_infer" attr, by default None.

        Returns
        -------
        xr.Dataset
            The Dataset containing the climatology departure between the
            original data variable and the data variable climatology.

        Examples
        --------
        Import:

        >>> import xarray as xr
        >>> from xcdat.climatology import climatology, departure

        Get departure for any time frequency:

        >>> ds = xr.open_dataset("file_path")
        >>> ts_climo = climatology(ds, "ts", "month")
        >>> ts_departure = departure(ds, ts_climo)

        Access attribute for info on departure operation:

        >>> ts_month_climo.operation
        {'type': 'departure', 'frequency': 'month', 'is_weighted': True}
        >>> ts_month_climo.attrs["operation"]
        {'type': 'departure', 'frequency': 'month', 'is_weighted': True}
        """
        # Use the climatology data variable to extract the climatology info.
        da_data_var = get_data_var(self._dataset, data_var)
        climo_info = da_data_var.attrs.get("operation")
        if climo_info is None:
            raise KeyError(
                f"The data var, '{da_data_var.name}', does not contain the "
                "'operation' attribute which describes the climatology operation. "
                f"Make sure to run the `ds.climo.cycle()` on '{da_data_var.name}' first "
                "before calculating its departure."
            )

        # Group the variable using the climatology information.
        data_var_og = self._dataset[f"{da_data_var.name}_original"]
        data_var_grouped = self._group_data(
            data_var_og,
            "departure",
            climo_info["frequency"],
            climo_info["is_weighted"],
            climo_info.get("djf_type"),
        ).rename("ts")

        # Calculate departure by subtracting the grouped data var with the
        # climatology data var.
        ds_departure = self._dataset.copy()
        with xr.set_options(keep_attrs=True):
            ds_departure[da_data_var.name] = data_var_grouped - da_data_var
        return ds_departure

    def _group_data(
        self,
        data_var: xr.DataArray,
        operation_type: Literal["climatology", "departure"],
        freq: Frequency,
        is_weighted: bool,
        djf_type: Optional[DJFType] = "cont",
    ) -> xr.DataArray:
        """Groups data variables by a frequency to get their averages.

        A Pandas MultiIndex is derived from the DateTime objects found in the
        "time" dimension. This allows for grouping based on multiple DateTime
        parameters (e.g., month and day), Lastly, information regarding the
        operation is added to the data variable for reference.

        Parameters
        ----------
        data_var : xr.DataArray
        The data variable to perform group operation on.
        operation_type : Literal["climatology", "departure"]
            The calculation type.
        freq : Frequency
            The frequency of time to group on.
        is_weighted : bool
            Perform grouping using weighted averages.
        djf_type : Optional[DJFType], optional
            Whether the DJF season is continuous (``"cont"``, previous year Dec)
            or discontinuous (``"discont"``, same year Dec), by default
            ``"cont"``.

        Returns
        -------
        xr.DataArray
            The grouped data variable
        """
        if is_weighted:
            weights = self.calculate_weights(data_var, freq) if is_weighted else None
            data_var *= weights

        if freq == "season" and djf_type == "cont":
            data_var = self._mask_incomplete_djf(data_var)

        data_var = self._groupby_multiindex(data_var, freq).sum()
        data_var = self._add_operation_attrs(
            data_var, operation_type, freq, is_weighted, djf_type
        )
        return data_var

    def calculate_weights(
        self, data_var: xr.DataArray, freq: Frequency
    ) -> xr.DataArray:
        """Calculates weights for a Dataset based on a frequency of time.

        Time bounds, leap years and number of days for each month are considered
        during grouping. If the sum of the weights does not equal 1.0, an error will
        be raised.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset used for calculating the lengths of months with bounds.
        data_var : xr.Dataset
            The data variable to calculate weights for.
        freq : Frequency
            The frequency of time to group on.
            Refer to ``FREQUENCIES`` for a complete list of available options.

        Returns
        -------
        xr.DataArray
            The weights based on a frequency of time.
        """
        months_lengths = self._get_months_lengths()
        months_lengths_grouped = self._groupby_multiindex(months_lengths, freq)

        weights: xr.DataArray = months_lengths_grouped / months_lengths_grouped.sum()
        self._validate_weights(data_var, weights, freq)
        return weights

    def _get_months_lengths(self) -> xr.DataArray:
        """Get lengths of months based on the time coordinates of a dataset.

        If time bounds exist, it will be used to generate the length of months. This
        allows for a robust calculation of weights because different datasets could
        record their time differently (e.g., at beginning/end/middle of each time
        interval).

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to get months' lengths from.

        Returns
        -------
        xr.DataArray
            The months' lengths for the dataset.
        """
        time_bounds = self._dataset.bounds.get_bounds("time")
        months_lengths = (time_bounds[:, 1] - time_bounds[:, 0]).dt.days
        return months_lengths

    def _validate_weights(
        self, data_var: xr.DataArray, weights: xr.DataArray, freq: Frequency
    ):
        """Validate that the sum of the weights for a dataset equals 1.0.

        It generates the number of frequency groups after grouping by a frequency.
        For example, if weights are being generated on a monthly basis, there are
        12 group with each group representing a month in the year.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable to validate weights for.
        weights : xr.DataArray
            The weights based on a frequency of time.
        datetime_component : DateTimeComponent
            The frequency of time to group by in xarray datetime component notation.
        """
        frequency_groups = len(self._groupby_multiindex(data_var, freq).count())
        expected_sum = np.ones(frequency_groups)
        actual_sum = self._groupby_multiindex(weights, freq).sum().values

        np.testing.assert_allclose(actual_sum, expected_sum)

    def _mask_incomplete_djf(self, data_var: xr.DataArray) -> xr.DataArray:
        """
        Masks the start year Jan/Feb and end year Dec, which are incomplete
        DJF seasons.

        These months are considered incomplete DJF seasons within the context of
        a continuous DJF, where the previous year December is included in the
        season rather than the same year.

        For example, (12/2001, 1/2002, 2/2002) is considered the continuous DJF
        season for 2002, not (1/2001, 2/2002, 12/2002).

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable to remove incomplete seasons from.

        Returns
        -------
        xr.DataArray
            The data variable with incomplete seasons removed
        """
        start_year = data_var.time.dt.year.data[0]
        end_year = data_var.time.dt.year.data[-1]

        incomplete_seasons = (f"{start_year}-01", f"{start_year}-02", f"{end_year}-12")
        for month in incomplete_seasons:
            try:
                data_var.sel(time=month)[:] = np.nan
            except KeyError:
                logging.debug("Dataset does not contain {month}.")

        return data_var

    def _groupby_multiindex(
        self, data_var: xr.DataArray, freq: Frequency
    ) -> xr.DataArray:
        """Groups a data variable by a pandas multiindex representing time.

        For example, if you are performing a daily climatology cycle calculation,
        the data must be grouped by the months and days.

        Exammple DateTime array representing "time" coordinates":
        ['1850-01-16T12:00:00.000000000', '1850-02-15T00:00:00.000000000',
        '1850-03-16T12:00:00.000000000'],

        Example resultant pandas multiindex array:
        [(01, 16), (02, 15), (03, 16)]

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable to perform grouping on.
        freq : Frequency
            The frequency to group on.

        Returns
        -------
        xr.DataArray
            The data variable grouped by the frequency.
        """
        data = []
        time_dim_key = data_var.cf["T"].name
        datetime_components = FREQS_TO_DATETIME[freq]

        for component in datetime_components:
            data.append(data_var[f"{time_dim_key}.{component}"].data)

        freq_index = pd.MultiIndex.from_arrays(data)
        coord_name = "_".join(datetime_components)
        data_var.coords[coord_name] = ("time", freq_index)
        data_var = data_var.groupby(coord_name)

        return data_var

    def _add_operation_attrs(
        self,
        data_var: xr.DataArray,
        operation_type: Literal["climatology", "departure"],
        frequency: Frequency,
        is_weighted: bool,
        djf_type: Optional[DJFType],
    ) -> xr.DataArray:
        """Adds operation attributes to the data var.

        These attributes should help users distinguish a data variable that
        has been operated on from their original counterpart.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.
        operation_type : Literal["climatology", "departure"],
            The calculation type.
        frequency : Frequency
            The frequency of time.
        is_weighted : bool
            Whether to calculation was weighted or not.
        djf_type : Optional[DJFType]
            Whether the DJF season is continuous ("cont", previous year Dec) or
            discontinuous ("discont", same year Dec), by default ``"cont"``.

        Returns
        -------
        xr.DataArray
            The data variable with a new `operation` attribute.

        Examples
        --------
        Access attribute for info on climatology operation:

        >>> ts_climo_monthly.operation
        {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
        >>> ts_climo_monthly.attrs["operation"]
        {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
        """
        data_var.attrs.update(
            {
                "operation": {
                    "type": operation_type,
                    "frequency": frequency,
                    "is_weighted": str(is_weighted),
                },
            }
        )

        if frequency == "season":
            data_var.attrs["operation"].update({"djf_type": djf_type})
        return data_var
