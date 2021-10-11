from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.climatology import DatasetClimatologyAccessor


class TestDatasetClimatologyAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

    def test__init__(self):
        obj = DatasetClimatologyAccessor(self.ds)

        assert obj._dataset.identical(self.ds)

    def test_decorator_call(self):
        obj = self.ds.climo

        assert obj._dataset.identical(self.ds)

    def test_raises_error_without_time_dimension(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")

        with pytest.raises(KeyError):
            ds.climo.cycle("season", "ts")

    def test_raises_error_with_incorrect_frequency_arg(self):
        with pytest.raises(ValueError):
            self.ds.climo.cycle("incorrect_frequency", data_var="ts")

    def test_raises_error_with_incorrect_djf_type_arg(self):
        with pytest.raises(ValueError):
            self.ds.climo.cycle(freq="season", data_var="ts", djf_type="incorrect")

    def test_raises_error_if_data_var_does_not_exist_in_dataset(self):
        with pytest.raises(KeyError):
            self.ds.climo.cycle(
                freq="season",
                data_var="non_existent_var",
            )

    def test_climatology_month_weighted_inferred_var(self):
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "ts"

        result_ds = self.ds.climo.cycle("month", data_var="ts")
        expected_ds = self.ds.copy()
        expected_ds["ts_original"] = expected_ds.ts.copy()
        expected_ds["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={
                "lat": self.ds.lat,
                "lon": self.ds.lon,
                "month": pd.MultiIndex.from_arrays(
                    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
                ),
            },
            dims=["month", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "climatology",
                    "frequency": "month",
                    "is_weighted": "True",
                }
            },
        )

        assert result_ds.identical(expected_ds)

    def test_climatology_month_weighted(self):
        result_ds = self.ds.climo.cycle("month", data_var="ts")

        expected_ds = self.ds.copy()
        expected_ds["ts_original"] = expected_ds.ts.copy()
        expected_ds["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={
                "lat": self.ds.lat,
                "lon": self.ds.lon,
                "month": pd.MultiIndex.from_arrays(
                    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
                ),
            },
            dims=["month", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "climatology",
                    "frequency": "month",
                    "is_weighted": "True",
                }
            },
        )

        assert result_ds.identical(expected_ds)

    def test_climatology_month_unweighted(self):
        result_ds = self.ds.climo.cycle("month", data_var="ts", is_weighted=False)

        expected_ds = self.ds.copy()
        expected_ds["ts_original"] = expected_ds.ts.copy()
        expected_ds["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={
                "lat": self.ds.lat,
                "lon": self.ds.lon,
                "month": pd.MultiIndex.from_arrays(
                    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
                ),
            },
            dims=["month", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "climatology",
                    "frequency": "month",
                    "is_weighted": "False",
                }
            },
        )
        expected_ds["ts"][0] = np.full((4, 4), [2.0, 2.0, 2.0, 2.0])
        expected_ds["ts"][1] = np.full((4, 4), [2.0, 2.0, 2.0, 2.0])

        assert result_ds.identical(expected_ds)


class TestDeparture:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        self.seasons = ["DJF", "JJA", "MAM", "SON"]

    def test_raises_error_if_climatology_was_not_run_on_the_data_var_first(self):
        with pytest.raises(KeyError):
            self.ds.climo.departure("ts")

    def test_weighted_seasonal_departure_with_continuous_djf_and_inferred_data_var(
        self,
    ):
        # Create a post-climatology dataset.
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "ts"
        ds["ts_original"] = ds.ts.copy()
        ds["ts"] = xr.DataArray(
            data=np.ones((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([self.seasons]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": "True",
                    "djf_type": "cont",
                },
            },
        )

        # Run climatology on the post-climatology dataset.
        result = ds.climo.departure()

        # Create an expected post-departure dataset.
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.zeros((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([self.seasons]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": "True",
                    "djf_type": "cont",
                },
            },
        )
        expected.ts.data[0] = [
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
        ]

        # Check all float values are close (raises error if not).
        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs

    def test_weighted_seasonal_departure_with_continuous_djf(self):
        # Create a post-climatology dataset.
        ds = self.ds.copy()
        ds["ts_original"] = ds.ts.copy()
        ds["ts"] = xr.DataArray(
            data=np.ones((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([self.seasons]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": "True",
                    "djf_type": "cont",
                },
            },
        )

        # Run climatology on the post-climatology dataset.
        result = ds.climo.departure("ts")

        # Create an expected post-departure dataset.
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.zeros((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([self.seasons]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": "True",
                    "djf_type": "cont",
                },
            },
        )
        expected.ts.data[0] = [
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
        ]

        # Check all float values are close (raises error if not).
        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs

    def test_unweighted_seasonal_departure_with_continuous_djf(self):
        # Create a post-climatology dataset.
        ds = self.ds.copy()
        ds["ts_original"] = ds.ts.copy()
        ds["ts"] = xr.DataArray(
            data=np.ones((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([self.seasons]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": "False",
                    "djf_type": "cont",
                },
            },
        )

        # Run climatology on the post-climatology dataset.
        result = ds.climo.departure("ts")

        # Create an expected post-departure dataset.
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.zeros((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([self.seasons]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": "False",
                    "djf_type": "cont",
                },
            },
        )
        expected.ts.data[0] = [
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
            [-0.39869281, -0.39869281, -0.39869281, -0.39869281],
        ]

        # Check all float values are close (raises error if not).
        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs

    def test_unweighted_seasonal_departure_with_dicontinuous_djf(self):
        # Create a post-climatology dataset.
        ds = self.ds.copy()
        ds["ts_original"] = ds.ts.copy()
        ds["ts"] = xr.DataArray(
            data=np.ones((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([self.seasons]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": "False",
                    "djf_type": "discont",
                },
            },
        )

        # Run climatology on the post-climatology dataset.
        result = ds.climo.departure("ts")

        # Create an expected post-departure dataset.
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.zeros((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([self.seasons]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": "False",
                    "djf_type": "discont",
                },
            },
        )

        assert result.identical(expected)


class TestGroupData:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)
        self.ds.attrs.update({"operation_type": "climatology"})

    def test__group_data_weighted_by_month_day(self):
        ds = self.ds.copy()
        ts_result = ds.climo._group_data(
            ds["ts"], "climatology", "day", is_weighted=True
        )
        assert ts_result.attrs["operation"]["type"] == "climatology"
        assert ts_result.attrs["operation"]["frequency"] == "day"
        assert ts_result.attrs["operation"]["is_weighted"] == "True"

        ts_expected = np.ones((12, 4, 4))
        assert np.allclose(ts_result, ts_expected)

    def test__group_data_unweighted_by_month_day(self):
        ds = self.ds.copy()
        ts_result = ds.climo._group_data(
            ds["ts"], "climatology", "day", is_weighted=False
        )
        assert ts_result.attrs["operation"]["type"] == "climatology"
        assert ts_result.attrs["operation"]["frequency"] == "day"
        assert ts_result.attrs["operation"]["is_weighted"] == "False"

        ts_expected = np.ones((12, 4, 4))
        ts_expected[0] = np.full((4, 4), [2, 2, 2, 2.0])
        ts_expected[1] = np.full((4, 4), [2, 2, 2, 2.0])

        assert np.allclose(ts_result, ts_expected)

    def test__group_data_weighted_by_month(self):
        ds = self.ds.copy()

        # Check non-bounds variables were properly grouped and averaged
        ts_result = ds.climo._group_data(
            ds["ts"], "climatology", "month", is_weighted=True
        )
        assert ts_result.attrs["operation"]["type"] == "climatology"
        assert ts_result.attrs["operation"]["frequency"] == "month"
        assert ts_result.attrs["operation"]["is_weighted"] == "True"

        ts_expected = np.ones((12, 4, 4))
        assert np.allclose(ts_result, ts_expected)

    def test__group_data_unweighted_by_month(self):
        ds = self.ds.copy()

        ts_result = ds.climo._group_data(
            ds["ts"], "climatology", "month", is_weighted=False
        )
        assert ts_result.attrs["operation"]["type"] == "climatology"
        assert ts_result.attrs["operation"]["frequency"] == "month"
        assert ts_result.attrs["operation"]["is_weighted"] == "False"

        ts_expected = np.ones((12, 4, 4))
        ts_expected[0] = np.full((4, 4), [2, 2, 2, 2.0])
        ts_expected[1] = np.full((4, 4), [2, 2, 2, 2.0])
        assert np.allclose(ts_result, ts_expected)

    def test__group_data_weighted_by_season_with_continuous_december(self):
        ds = self.ds.copy()

        # Check non-bounds variables were properly grouped and averaged
        ts_result = ds.climo._group_data(
            ds["ts"],
            "climatology",
            "season",
            is_weighted=True,
            djf_type="cont",
        )
        assert ts_result.attrs["operation"]["type"] == "climatology"
        assert ts_result.attrs["operation"]["frequency"] == "season"
        assert ts_result.attrs["operation"]["is_weighted"] == "True"

        ts_expected = np.ones((4, 4, 4))
        ts_expected[0] = [
            [0.60130719, 0.60130719, 0.60130719, 0.60130719],
            [0.60130719, 0.60130719, 0.60130719, 0.60130719],
            [0.60130719, 0.60130719, 0.60130719, 0.60130719],
            [0.60130719, 0.60130719, 0.60130719, 0.60130719],
        ]
        assert np.allclose(ts_result, ts_expected)

    def test__group_data_unweighted_by_season_with_continuous_december(self):
        ds = self.ds.copy()

        ts_result = ds.climo._group_data(
            ds["ts"],
            "climatology",
            "season",
            is_weighted=False,
            djf_type="cont",
        )
        assert ts_result.attrs["operation"]["type"] == "climatology"
        assert ts_result.attrs["operation"]["frequency"] == "season"
        assert ts_result.attrs["operation"]["is_weighted"] == "False"

        ts_expected = np.full((4, 4, 4), 3)
        assert np.allclose(ts_result, ts_expected)

    def test__group_data_weighted_by_season_with_discontinuous_december(self):
        ds = self.ds.copy()

        ts_result = ds.climo._group_data(
            ds["ts"],
            "climatology",
            "season",
            is_weighted=True,
            djf_type="discont",
        )
        assert ts_result.attrs["operation"]["type"] == "climatology"
        assert ts_result.attrs["operation"]["frequency"] == "season"
        assert ts_result.attrs["operation"]["is_weighted"] == "True"

        ts_expected = np.ones((4, 4, 4))
        assert np.allclose(ts_result, ts_expected)

    def test__group_data_unweighted_by_season_with_discontinuous_december(self):
        ds = self.ds.copy()

        ts_result = ds.climo._group_data(
            ds["ts"],
            "climatology",
            "season",
            is_weighted=False,
            djf_type="discont",
        )
        assert ts_result.attrs["operation"]["type"] == "climatology"
        assert ts_result.attrs["operation"]["frequency"] == "season"
        assert ts_result.attrs["operation"]["is_weighted"] == "False"

        ts_expected = np.full((4, 4, 4), 3.0)
        ts_expected[0] = np.full((4, 4), [5.0, 5.0, 5.0, 5.0])

        assert np.allclose(ts_result, ts_expected)


class TestRemoveIncompleteDJF:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_incomplete_djf_seasons_are_masked(self):
        ts = xr.DataArray(
            data=np.ones(5),
            coords={
                "time": [
                    datetime(2000, 1, 1),
                    datetime(2000, 2, 1),
                    datetime(2000, 3, 1),
                    datetime(2000, 4, 1),
                    datetime(2001, 12, 1),
                ]
            },
            dims=["time"],
        )

        result = self.ds.climo._mask_incomplete_djf(ts)
        expected = ts.copy()
        # Mask the start year Jan/Feb and end year Dec
        expected.data[0] = np.nan
        expected.data[1] = np.nan
        expected.data[-1] = np.nan

        assert result.identical(expected)

    def test_does_not_mask_incomplete_seasons_dont_exist(self):
        ts = xr.DataArray(
            data=np.ones(5),
            coords={
                "time": [
                    datetime(2000, 3, 1),
                    datetime(2000, 4, 1),
                    datetime(2000, 5, 1),
                    datetime(2000, 6, 1),
                    datetime(2000, 7, 1),
                ]
            },
            dims=["time"],
        )

        result = self.ds.climo._mask_incomplete_djf(ts)
        expected = ts

        assert result.identical(expected)


class TestCalculateWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

    def test__get_months_lengths_raises_error_without_time_bounds(self):
        ds = self.ds.copy()
        ds = ds.drop_vars({"time_bnds"})

        with pytest.raises(KeyError):
            ds.climo._get_months_lengths()

    def test_calculate_weights_daily(self):
        expected = np.array(
            [
                0.5,
                0.49180328,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.50819672,
            ]
        )
        result = self.ds.climo.calculate_weights(self.ds["ts"], "day")
        assert np.allclose(result, expected)

    def test_calculate_weights_monthly(self):
        # TODO:
        expected = np.array(
            [
                0.5,
                0.49180328,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.50819672,
            ]
        )
        result = self.ds.climo.calculate_weights(self.ds["ts"], "month")
        assert np.allclose(result, expected)

    def test_calculate_weights_seasonal(self):
        expected = np.array(
            [
                0.20261438,
                0.19607843,
                0.33333333,
                0.33333333,
                0.33333333,
                0.32967033,
                0.32967033,
                0.34065934,
                0.33333333,
                0.33333333,
                0.33333333,
                0.19607843,
                0.20261438,
                0.20261438,
            ]
        )
        result = self.ds.climo.calculate_weights(self.ds["ts"], "season")
        assert np.allclose(result, expected)


class TestAddAttributes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_adds_attrs_to_data_var(self):
        ts = self.ds.ts

        result_ts = self.ds.climo._add_operation_attrs(
            ts,
            operation_type="climatology",
            frequency="season",
            is_weighted=True,
            djf_type="cont",
        )
        expected_ts = ts.copy()
        expected_ts.attrs.update(
            {
                "operation": {
                    "type": "climatology",
                    "frequency": "season",
                    "is_weighted": "True",
                    "djf_type": "cont",
                }
            }
        )

        assert result_ts.identical(expected_ts)
