"""Module containing temporal averaging (time series, climatology) functions."""

from typing import Optional, Union

import cf_xarray as cfxr  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Literal, get_args

from xcdat import bounds, logger  # noqa: F401
from xcdat.dataset import get_data_var

logging = logger.setup_custom_logger("root")

# Time Averaging Operations
# =========================
# Type alias for time averaging operations.
Operation = Literal["climatology", "departure", "timeseries_avg"]

# Frequencies
# ===========
# Type alias for all available frequencies.
Frequency = Union[Literal["day", "month", "season", "year", "month"]]
#: Tuple of available frequencies for the ``frequency`` param.
FREQUENCIES = ("hour", "day", "month", "season", "year", "month")

# Datetime Components
# ===================
# Type alias representing xarray DateTime components.
DateTimeComponent = Literal["hour", "day", "month", "season", "year"]

# DJF Season Types
# ================
# Type alias for the DJF season being continuous or discontinuous.
DJFType = Literal["cont", "discont"]
# Tuple of DJF season type params.
DJF_TYPES = get_args(DJFType)


class TemporalAverageAccessor:
    # Maps frequencies to xarray DateTime components, which are used to create
    # a Pandas MultiIndex for time grouping operations.
    # Source: https://xarray.pydata.org/en/stable/user-guide/time-series.html#datetime-components
    FREQ_GROUPBY_MAP = {
        # TODO: Add support for custom seasons freq
        # TODO: Add support for diurnalNNN freq
        "climatology": {
            # For "season" frequency, "year and "month" must be included to
            # properly mask incomplete seasons and shift over Decembers by year
            # (for continuous DJF).
            # Both index levels are removed from the MultiIndex before grouping
            # by it. Refer to `_drop_season_multiindex_levels()`.
            "season": ("year", "season", "month"),
            "month": ("month",),
            "day": ("month", "day"),
        },
        "departure": {
            "season": ("year", "season", "month"),
            "month": ("month",),
            "day": ("month", "day"),
        },
        # TODO: Add support for custom seasons
        # TODO: Add support for Nhour
        "timeseries_avg": {
            "year": ("year",),
            # For the "seasona" freq, "month" must be included to properly mask
            # incomplete seasons and shift over Decembers by year
            # (for continuous DJF).
            # The "month" index level is removed from the MultiIndex before
            # grouping by it. Refer to `_drop_season_multiindex_levels()`.
            "season": ("year", "season", "month"),
            "month": ("year", "month"),
            "day": ("year", "month", "day"),
            "hour": ("year", "month", "day", "hour"),
        },
    }

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

        # The time averaging operation performed.
        self.operation: Operation
        # The frequency of time to group by.
        self.freq: Frequency
        # Perform grouping using weighted averages, by default True.
        self.is_weighted: bool = True
        # Whether the DJF season is continuous ("cont", previous year Dec) or
        # discontinuous ("discont", same year Dec), by default ``"cont"``.
        self.djf_type: DJFType = "cont"
        # A Pandas MultiIndex created from the time dimension for use in
        # group by operations. Using a MultiIndex allows grouping by multiple
        # DateTime components, such as year and month (native xarray does not
        # support this).
        self.time_multiindex: pd.MultiIndex
        self.time_multiindex_name: str

    def _validate_and_set_attrs(
        self,
        operation: Operation,
        freq: Frequency,
        is_weighted: bool,
        djf_type: DJFType,
    ):
        """Validates inputs and sets their equivalent object attribute.

        Parameters
        ----------
        operation: Operation
            The time averaging operation being performed.
        freq : Frequency
            The frequency of time to group by.
        is_weighted: bool
            Perform weighted or unweighted averages.
        djf_type : DJFType
            Whether the DJF season is "cont" (continuous with previous year
            December) or "discont" (discontinuous with same year Dec).

        Raises
        ------
        KeyError
            If the Dataset does not have a "time" dimension.
        ValueError
            If an incorrect ``freq`` arg was passed.
        ValueError
            If an incorrect ``djf_type`` arg was passed.
        """
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

        self.operation = operation
        self.freq = freq
        self.is_weighted = is_weighted
        self.djf_type = djf_type

    def _group_data(self, data_var: xr.DataArray) -> xr.DataArray:
        """Groups data variables by a frequency to get their averages.

        A Pandas MultiIndex is derived from the DateTime64 objects found in the
        "time" coordinates. Using the Pandas MultiIndex allows for grouping
        based on multiple DateTime parameters, such as month and day.

        Once the grouping operation is completed, the parameters of the
        operation are stored within the data variable as an attribute.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable to perform grouping operation on.

        Returns
        -------
        xr.DataArray
            The grouped data variable
        """
        self._set_time_multiindex(data_var)

        if self.freq == "season" and self.djf_type == "cont":
            data_var = self._mask_incomplete_djf(data_var)

        if self.is_weighted:
            weights = self.calculate_weights(data_var)
            data_var *= weights
            data_var = self._time_multiindex_name(data_var).sum()
        else:
            data_var = self._time_multiindex_name(data_var).mean()

        data_var = self._add_operation_attrs(data_var)
        return data_var

    def _set_time_multiindex(self, data_var: xr.DataArray):
        """
        Creates a Pandas MultiIndex from the time coordinates and sets the
        related object attributes.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.
        """
        time_dim_key = data_var.cf["T"].name
        datetime_components = TemporalAverageAccessor.FREQ_GROUPBY_MAP[self.operation][
            self.freq
        ]

        # A DataFrame to store the DateTime components extracted from
        # each object in the time coordinates. It is used to generate
        # a Pandas MultiIndex for xarray grouping operations.
        df = pd.DataFrame()
        for component in datetime_components:
            df[component] = data_var[f"{time_dim_key}.{component}"].data

        if self.freq == "season":
            if self.djf_type == "cont":
                df = self._shift_decembers(df)
            df = self._drop_multiindex_levels(df)
        self.df = df

        # `pd.MultiIndex.from_frame()` generates the `names` of the MultiIndex
        # using the DataFrame column names. However, this default behavior
        # clashes with xarray because xarray also generates the `names` of the
        # MultiIndex when assigning it to the data variable's coordinates.
        # Xarray sees that the same `names` exist already, so it will throw
        # `ValueError: conflicting MultiIndex level name(s):``.
        # The workaround is to assign a placeholder for `names`, which will be
        # overwritten by xarray once the MultiIndex is assigned to the
        # data variable's coordinates.
        # Related issue: https://github.com/pydata/xarray/issues/3659
        placeholder_names = [(index, col) for index, col in enumerate(df.columns)]
        self.time_multiindex = pd.MultiIndex.from_frame(df, names=placeholder_names)
        self.time_multiindex_name = "_".join(df.columns)

    def _shift_decembers(self, df_time: pd.DataFrame) -> pd.DataFrame:
        """Shifts Decembers over to the next year for continuous DJF seasons.

        Xarray defines the DJF season with the same year December, resulting in
        discontinuous DJF seasons. If the intent is to have continuous DJF
        seasons, the Decembers should be from the previous years.

        Parameters
        ----------
        df_time : pd.DataFrame
            The DataFrame generated from the time coordinates, with each column
            storing the xarray DateTime component values.

        Returns
        -------
        pd.DataFrame
            The DataFrame with Decembers shifted over year.
        """
        df_time.loc[df_time["month"] == 12, "year"] = df_time["year"] + 1
        return df_time

    def _drop_multiindex_levels(self, df_time: pd.DataFrame) -> pd.DataFrame:
        """Drops columns from the season DataFrame based on the operation.

        Specific columns are dropped because they are no longer necessary for
        the creation of the seasonal MultiIndex.

        Parameters
        ----------
        df_time : pd.DataFrame
            The DataFrame generated from the time coordinates, with each column
            storing the xarray DateTime component values.

        Returns
        -------
        pd.DataFrame
            The DataFrame with droppped levels.
        """
        if self.operation == "timeseries_avg":
            df_time = df_time.drop("month", axis=1)
        elif self.operation in ["climatology", "departure"]:
            df_time = df_time.drop(["year", "month"], axis=1)
        return df_time

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

    def calculate_weights(self, data_var: xr.DataArray) -> xr.DataArray:
        """Calculates weights for a Dataset based on a frequency of time.

        Time bounds, leap years and number of days for each month are considered
        during grouping. If the sum of the weights do not equal 1.0, an error
        will be raised.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset used for calculating the lengths of months with bounds.

        Returns
        -------
        xr.DataArray
            The weights based on a frequency of time.
        """
        months_lengths = self._get_months_lengths()
        months_lengths_grouped = self._time_multiindex_name(months_lengths)

        weights: xr.DataArray = months_lengths_grouped / months_lengths_grouped.sum()
        self._validate_weights(data_var, weights)
        return weights

    def _get_months_lengths(self) -> xr.DataArray:
        """Get lengths of months based on the time coordinates of a dataset.

        If time bounds exist, it will be used to generate the length of months.
        This allows for a robust calculation of weights because different
        datasets could record their time differently (e.g., at the beginning,
        middle, or end of each time interval).

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

    def _validate_weights(self, data_var: xr.DataArray, weights: xr.DataArray):
        """Validate that the sum of the weights for a dataset equals 1.0.

        It generates the number of frequency groups after grouping by a
        frequency. For example, if weights are being generated on a monthly
        basis, there are 12 group with each group representing a month in the
        year.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable to validate weights for.
        """
        frequency_groups = len(self._time_multiindex_name(data_var).count())
        expected_sum = np.ones(frequency_groups)
        actual_sum = self._time_multiindex_name(weights).sum().values

        np.testing.assert_allclose(actual_sum, expected_sum)

    def _time_multiindex_name(self, data_var: xr.DataArray) -> xr.DataArray:
        """Groups a data variable by a Pandas MultiIndex for time coordinates.

        For example, if you are performing a daily climatology cycle
        calculation, the data must be grouped by the months and days.

        Exammple DateTime array representing "time" coordinates":
        ['1850-01-16T12:00:00.000000000', '1850-02-15T00:00:00.000000000',
        '1850-03-16T12:00:00.000000000'],

        Example resultant pandas multiindex array:
        [(01, 16), (02, 15), (03, 16)]

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable to perform grouping on.

        Returns
        -------
        xr.DataArray
            The data variable grouped by the frequency.
        """
        data_var.coords[self.time_multiindex_name] = (
            data_var.cf["T"].name,
            self.time_multiindex,
        )
        data_var = data_var.groupby(self.time_multiindex_name)

        return data_var

    def _add_operation_attrs(self, data_var: xr.DataArray) -> xr.DataArray:
        """Adds operation attributes to the data var.

        These attributes should help users distinguish a data variable that
        has been operated on from their original counterpart.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.

        Returns
        -------
        xr.DataArray
            The data variable with a new `operation` attribute.

        Examples
        --------
        Access attribute for info on climatology operation:

        >>> ds_climo_seasonal.ts.operation
        {'type': 'climatology', 'freq': 'month', 'is_weighted': True
         'groupby': 'season'
        }
        >>> ds_ts_seasonal.ts.attrs["operation"]
        {'type': 'timeseries_avg', 'freq': 'season', 'is_weighted': True
         'groupby':  'year_season'
        }
        """
        data_var.attrs.update(
            {
                "operation": {
                    "type": self.operation,
                    "freq": self.freq,
                    "groupby": self.time_multiindex_name,
                    "is_weighted": str(self.is_weighted),
                },
            }
        )

        if self.freq == "season":
            data_var.attrs["operation"].update({"djf_type": self.djf_type})
        return data_var


@xr.register_dataset_accessor("climo")
class ClimatologyAccessor(TemporalAverageAccessor):
    def __init__(self, dataset: xr.Dataset):
        super(ClimatologyAccessor, self).__init__(dataset)

    def cycle(
        self,
        freq: Frequency,
        data_var: Optional[str] = None,
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
        djf_type : DJFType
            Whether the DJF season is "cont" (continuous with previous year
            December) or "discont" (discontinuous with same year Dec), by
            default ``"cont"``.

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
        self._validate_and_set_attrs("climatology", freq, is_weighted, djf_type)
        ds_climo = self._dataset.copy()
        da_data_var = get_data_var(ds_climo, data_var)

        # Calculate data variable climatology and preserve the original variable
        # for calculating departure.
        ds_climo[da_data_var.name] = self._group_data(da_data_var.copy())
        ds_climo[f"{da_data_var.name}_original"] = da_data_var.copy()
        return ds_climo

    def departure(self, data_var: Optional[str] = None) -> xr.Dataset:
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
        # The climatology operation parameters are extracted and used for
        # the departure operation.
        da_data_var = get_data_var(self._dataset, data_var)
        climo_info = da_data_var.attrs.get("operation")
        if climo_info is None:
            raise KeyError(
                f"The data var, '{da_data_var.name}', does not contain the "
                "'operation' attribute which describes the climatology operation. "
                f"Make sure to run the `ds.climo.cycle()` on '{da_data_var.name}' first "
                "before calculating its departure."
            )
        self._validate_and_set_attrs(
            "departure",
            climo_info["freq"],
            climo_info["is_weighted"],
            climo_info.get("djf_type", "cont"),
        )

        data_var_og = self._dataset[f"{da_data_var.name}_original"].copy()
        data_var_grouped = self._group_data(data_var_og).rename("ts")

        # Calculate departure by subtracting the grouped data var with the
        # climatology data var.
        ds_departure = self._dataset.copy()
        with xr.set_options(keep_attrs=True):
            ds_departure[da_data_var.name] = data_var_grouped - da_data_var
        return ds_departure


@xr.register_dataset_accessor("timeseries")
class TimeseriesAverageAccessor(TemporalAverageAccessor):
    """Class to represent TimeSeriesAverageAccessor"""

    def __init__(self, dataset: xr.Dataset):
        super(TimeseriesAverageAccessor, self).__init__(dataset)

    def avg(
        self,
        freq: Frequency,
        data_var: Optional[str] = None,
        is_weighted: bool = True,
        djf_type: DJFType = "cont",
    ) -> xr.Dataset:
        """Calculates the timeseries average for a data variable.

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
        djf_type : DJFType
            Whether the DJF season is "cont" (continuous with previous year
            December) or "discont" (discontinuous with same year Dec), by
            default ``"cont"``.

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

        >>> ds_ts_daily = ds.timeseries.avg("day", data_var="ts")
        >>> ds_ts_daily.ts
        >>>
        >>> ds_ts_seasonal = ds.timeseries.avg("season", data_var="ts")
        >>> ds_ts_seasonal.ts
        >>>
        >>> ds_ts_annual = ds.timeseries.avg("year", data_var="ts")
        >>> ds_ts_annual.ts

        Get monthly, seasonal, or month unweighted climatology for a variable:
        >>> ds_ts_daily = ds.timeseries.avg("day", data_var="ts", is_weighted=False)
        >>> ds_ts_daily.ts
        >>>
        >>> ds_ts_seasonal = ds.timeseries.avg("season", data_var="ts", is_weighted=False)
        >>> ds_ts_seasonal.ts
        >>>
        >>> ds_ts_annual = ds.timeseries.avg("year", data_var="ts", is_weighted=False)
        >>> ds_ts_annual.ts


        Access Dataset attribute for climatology operation info:

        >>> ds_ts_monthly.ts.operation
        {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
        >>> ds_ts_monthly.ts.attrs["operation"]
        {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
        """
        self._validate_and_set_attrs("timeseries_avg", freq, is_weighted, djf_type)

        ds = self._dataset.copy()
        da_data_var = get_data_var(ds, data_var)

        ds[da_data_var.name] = self._group_data(da_data_var.copy())
        return ds
