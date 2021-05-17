import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset, lat_bnds, lon_bnds, time_bnds
from xcdat.bounds import DatasetBoundsAccessor


class TestDatasetBoundsAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test__init__(self):
        obj = DatasetBoundsAccessor(self.ds)
        assert obj._dataset.identical(self.ds)

    def test_decorator_call(self):
        assert self.ds.bounds._dataset.identical(self.ds)

    def test_bounds_property_returns_map_of_coordinate_key_to_bounds_dataarray(self):
        ds = self.ds_with_bnds.copy()
        expected = {
            "T": ds.time_bnds,
            "X": ds.lon_bnds,
            "Y": ds.lat_bnds,
            "lat": ds.lat_bnds,
            "latitude": ds.lat_bnds,
            "lon": ds.lon_bnds,
            "longitude": ds.lon_bnds,
            "time": ds.time_bnds,
        }

        result = ds.bounds.bounds

        for key in expected.keys():
            assert result[key].identical(expected[key])

    def test_names_property_returns_a_list_of_sorted_bounds_names(self):
        result = self.ds_with_bnds.bounds.names
        expected = ["lat_bnds", "lon_bnds", "time_bnds"]

        assert result == expected

    def test_fill_missing_returns_dataset_with_filled_bounds(self):
        ds = self.ds_with_bnds.copy()

        ds = ds.drop_vars(["lat_bnds", "lon_bnds"])

        result = ds.bounds.fill_missing()
        assert result.identical(self.ds_with_bnds)

    def test_get_bounds_returns_error_when_bounds_dont_exist(self):
        with pytest.raises(KeyError):
            self.ds.bounds.get_bounds("lat")

    def test_get_bounds_when_bounds_exist_in_dataset(self):
        ds = self.ds_with_bnds.copy()
        lat_bnds = ds.bounds.get_bounds("lat")
        assert lat_bnds.identical(ds.lat_bnds)

        lon_bnds = ds.bounds.get_bounds("lon")
        assert lon_bnds.identical(ds.lon_bnds)
        assert lon_bnds.is_generated

    def test_get_bounds_when_bounds_do_not_exist_in_dataset(self):
        ds = self.ds_with_bnds.copy()

        with pytest.raises(KeyError):
            ds = ds.drop_vars(["lat_bnds"])
            ds.bounds.get_bounds("lat")

    def test_get_bounds_raises_error_with_incorrect_coord_argument(self):
        with pytest.raises(ValueError):
            self.ds.bounds.get_bounds("incorrect_coord_argument")

    def test_add_bounds_raises_error_if_bounds_exist(self):
        ds = self.ds_with_bnds.copy()

        with pytest.raises(ValueError):
            ds.bounds.add_bounds("lat")

    def test__add_bounds_raises_errors_for_data_dim_and_length(self):
        # Multidimensional
        lat = xr.DataArray(
            data=np.array([[0, 1, 2], [3, 4, 5]]),
            dims=["placeholder_1", "placeholder_2"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        # Length <=1
        lon = xr.DataArray(
            data=np.array([0]),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        # If coords dimensions does not equal 1.
        with pytest.raises(ValueError):
            ds.bounds._add_bounds("lat")
        # If coords are length of <=1.
        with pytest.raises(ValueError):
            ds.bounds._add_bounds("lon")

    def test__add_bounds_returns_dataset_with_bounds_added(self):
        ds = self.ds.copy()

        ds = ds.bounds._add_bounds("lat")
        assert ds.lat_bnds.equals(lat_bnds)
        assert ds.lat_bnds.is_generated == "True"

        ds = ds.bounds._add_bounds("lon")
        assert ds.lon_bnds.equals(lon_bnds)
        assert ds.lon_bnds.is_generated == "True"

        ds = ds.bounds._add_bounds("time")
        assert ds.time_bnds.equals(time_bnds)
        assert ds.time_bnds.is_generated == "True"

    def test__get_coord(self):
        ds = self.ds.copy()

        # Check lat axis coordinates exist
        lat = ds.bounds._get_coord("lat")
        assert lat is not None

        # Check lon axis coordinates exist
        lon = ds.bounds._get_coord("lon")
        assert lon is not None

    def test__get_coord_raises_error_if_coord_does_not_exist(self):
        ds = self.ds.copy()

        ds = ds.drop_dims("lat")
        with pytest.raises(KeyError):
            ds.bounds._get_coord("lat")
