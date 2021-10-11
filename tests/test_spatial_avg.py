import numpy as np
import pytest
import xarray as xr

import xcdat.spatial_avg  # noqa: F401
from tests.fixtures import generate_dataset


class TestSpatialAverage:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

        # Limit to just 3 data points to simplify testing.
        self.ds = self.ds.isel(time=slice(None, 3))

        # Change the value of the first element so that it is easier to identify
        # changes in the output.
        self.ds["ts"].data[0] = np.full((4, 4), 2.25)

    def test_raises_error_if_data_var_not_in_dataset(self):
        with pytest.raises(KeyError):
            self.ds.spatial.avg(
                "not_a_data_var",
                axis=["lat", "incorrect_axess"],
            )

    def test_weighted_spatial_average_for_lat_and_lon_region_for_an_inferred_data_var(
        self,
    ):
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "ts"

        # `data_var` kwarg is not specified, so an inference is attempted
        result = ds.spatial.avg(
            axis=["lat", "lon"], lat_bounds=(-5.0, 5), lon_bounds=(-170, -120.1)
        )

        expected = self.ds.copy()
        expected.attrs["xcdat_infer"] = "ts"
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        assert result.identical(expected)

    def test_weighted_spatial_average_for_lat_and_lon_region_for_explicit_data_var(
        self,
    ):
        ds = self.ds.copy()
        result = ds.spatial.avg(
            "ts", axis=["lat", "lon"], lat_bounds=(-5.0, 5), lon_bounds=(-170, -120.1)
        )

        expected = self.ds.copy()

        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        assert result.identical(expected)

    def test_weighted_spatial_average_for_lat_region(self):
        ds = self.ds.copy()

        # Specifying axis as a str instead of list of str.
        result = ds.spatial.avg(
            "ts", axis="lat", lat_bounds=(-5.0, 5), lon_bounds=(-170, -120.1)
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array(
                [[2.25, 2.25, 2.25, 2.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            coords={"time": expected.time, "lon": expected.lon},
            dims=["time", "lon"],
        )

        assert result.identical(expected)


class TestValidateAxis:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_axis_list_contains_unsupported_axes(self):
        with pytest.raises(ValueError):
            self.ds.spatial._validate_axis(self.ds.ts, axis=["lat", "incorrect_axes"])

    def test_raises_error_if_lat_axes_does_not_exist(self):
        ds = self.ds.copy()
        ds["ts"] = xr.DataArray(data=None, coords={"lon": ds.lon}, dims=["lon"])
        with pytest.raises(KeyError):
            ds.spatial._validate_axis(ds.ts, axis=["lat", "lon"])

    def test_raises_error_if_lon_axes_does_not_exist(self):
        ds = self.ds.copy()
        ds["ts"] = xr.DataArray(data=None, coords={"lat": ds.lat}, dims=["lat"])
        with pytest.raises(KeyError):
            ds.spatial._validate_axis(ds.ts, axis=["lat", "lon"])

    def test_returns_list_of_str_if_axis_is_a_single_supported_str_input(self):
        result = self.ds.spatial._validate_axis(self.ds.ts, axis="lat")
        expected = ["lat"]
        assert result == expected


class TestValidateRegionBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_bounds_type_is_not_a_tuple(self):
        with pytest.raises(TypeError):
            self.ds.spatial._validate_region_bounds("lon", [1, 1])

        with pytest.raises(TypeError):
            self.ds.spatial._validate_region_bounds("lon", "str")

    def test_raises_error_if_there_are_0_elements_in_the_bounds(self):
        with pytest.raises(ValueError):
            self.ds.spatial._validate_region_bounds("lon", ())

    def test_raises_error_if_there_are_more_than_two_elements_in_the_bounds(self):
        with pytest.raises(ValueError):
            self.ds.spatial._validate_region_bounds("lon", (1, 1, 2))

    def test_does_not_raise_error_if_lower_and_upper_bounds_are_floats_or_ints(self):
        self.ds.spatial._validate_region_bounds("lon", (1, 1))
        self.ds.spatial._validate_region_bounds("lon", (1, 1.2))

    def test_raises_error_if_lower_bound_is_not_a_float_or_int(self):
        with pytest.raises(TypeError):
            self.ds.spatial._validate_region_bounds("lat", ("invalid", 1))

    def test_raises_error_if_upper_bound_is_not_a_float_or_int(self):
        with pytest.raises(TypeError):
            self.ds.spatial._validate_region_bounds("lon", (1, "invalid"))

    def test_raises_error_if_lower_lat_bound_is_bigger_than_upper(self):
        with pytest.raises(ValueError):
            self.ds.spatial._validate_region_bounds("lat", (2, 1))

    def test_does_not_raise_error_if_lon_lower_bound_is_larger_than_upper(self):
        self.ds.spatial._validate_region_bounds("lon", (2, 1))


class TestValidateWeights:
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)
        self.weights = xr.DataArray(
            data=np.ones((4, 4)),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

    def test_no_error_is_raised_when_spatial_dim_sizes_align_between_weights_and_data_var(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones((4, 4)),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )
        self.ds.spatial._validate_weights(self.ds["ts"], axis="lat", weights=weights)

    def test_error_is_raised_when_lat_axis_is_specified_but_lat_is_not_in_weights_dims(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones(4), coords={"lon": self.ds.lon}, dims=["lon"]
        )
        with pytest.raises(KeyError):
            self.ds.spatial._validate_weights(
                self.ds["ts"], axis=["lon", "lat"], weights=weights
            )

    def test_error_is_raised_when_lon_axis_is_specified_but_lon_is_not_in_weights_dims(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones(4), coords={"lat": self.ds.lat}, dims=["lat"]
        )
        with pytest.raises(KeyError):
            self.ds.spatial._validate_weights(
                self.ds["ts"], axis=["lon", "lat"], weights=weights
            )

    def test_error_is_raised_when_weights_lat_and_lon_dims_dont_align_with_data_var_dims(
        self,
    ):
        # Get a slice of the dataset to reduce the size of the dimensions for
        # simpler testing.
        ds = self.ds.isel(lat=slice(0, 3), lon=slice(0, 3))
        weights = xr.DataArray(
            data=np.ones((3, 3)),
            coords={"lat": ds.lat, "lon": ds.lon},
            dims=["lat", "lon"],
        )

        with pytest.raises(ValueError):
            self.ds.spatial._validate_weights(
                self.ds["ts"], axis=["lat", "lon"], weights=weights
            )


class TestSwapRegionLonAxis:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_successful_swap_from_180_to_360(self):
        result = self.ds.spatial._swap_lon_axes(np.array([-65, 0, 120]), to="360")
        expected = np.array([295, 0, 120])

        assert np.array_equal(result, expected)

        expected = np.array([180, 0, 180])
        result = self.ds.spatial._swap_lon_axes(np.array([-180, 0, 180]), to="360")

        assert np.array_equal(result, expected)

    def test_successful_swap_from_360_to_180(self):
        result = self.ds.spatial._swap_lon_axes(np.array([0, 120, 181, 360]), to="180")
        expected = np.array([0, 120, -179, 0])

        assert np.array_equal(result, expected)

        result = self.ds.spatial._swap_lon_axes(
            np.array([-0.25, 120, 359.75]), to="180"
        )
        expected = np.array([-0.25, 120, -0.25])

        assert np.array_equal(result, expected)


class TestGetWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_returns_area_weights_for_region_within_lat_and_lon(self):
        result = self.ds.spatial._get_weights(
            axis=["lat", "lon"], lat_bounds=(-5, 5), lon_bounds=(-170, -120)
        )
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.35778714, 0.0],
                    [0.0, 0.0, 4.35778714, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )
        xr.testing.assert_allclose(result, expected)

    def test_returns_area_weights_for_region_within_lat(self):
        result = self.ds.spatial._get_weights(
            axis=["lat", "lon"], lat_bounds=(-5, 5), lon_bounds=None
        )
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.16341702, 15.52461668, 15.52461668, 0.16341702],
                    [0.16341702, 15.52461668, 15.52461668, 0.16341702],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_returns_area_weights_for_region_within_lon(self):
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.00297475, 0.0],
                    [0.0, 0.0, 49.99702525, 0.0],
                    [0.0, 0.0, 49.99702525, 0.0],
                    [0.0, 0.0, 0.00297475, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )
        result = self.ds.spatial._get_weights(
            axis=["lat", "lon"], lat_bounds=None, lon_bounds=(-170, -120)
        )

        xr.testing.assert_allclose(result, expected)


class TestForceLonLinearity:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_forces_linearity_for_rows_in_domain_bounds_where_lower_val_is_larger_than_upper(
        self,
    ):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [-0.9375, 0.9375],
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lon": self.ds.lon},
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        result_domain, result_region = self.ds.spatial._force_lon_linearity(
            domain_bounds=domain_bounds, region_bounds=np.array([5.0, -5.0])
        )
        expected_domain = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [359.0625, 360.9375],
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            dims=["lon", "bnds"],
            coords={"lon": self.ds.lon},
            attrs={"is_generated": "True"},
        )
        expected_region = np.array([5.0, 355.0])

        assert result_domain.identical(expected_domain)
        assert np.array_equal(result_region, expected_region)

    def test_forces_linearity_for_values_in_region_bounds_that_are_less_than_domain_lower_bound(
        self,
    ):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                    [359.0625, 360.9375],
                ]
            ),
            dims=["lon", "bnds"],
            coords={"lon": self.ds.lon},
            attrs={"is_generated": "True"},
        )
        result_domain, result_region = self.ds.spatial._force_lon_linearity(
            domain_bounds=domain_bounds, region_bounds=np.array([0.0, 5.0])
        )
        expected_region = np.array([360, 5.0])

        assert result_domain.identical(domain_bounds)
        assert np.array_equal(result_region, expected_region)

    def test_does_not_update_already_linear_longitude_bounds(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                    [359.0625, 360.9375],
                ]
            ),
            dims=["lon", "bnds"],
            coords={"lon": self.ds.lon},
            attrs={"is_generated": "True"},
        )
        region_bounds = np.array([0.9375, 5.0])

        result_domain, result_region = self.ds.spatial._force_lon_linearity(
            domain_bounds=domain_bounds, region_bounds=region_bounds
        )

        assert result_domain.identical(domain_bounds)
        assert np.array_equal(result_region, region_bounds)


class TestSetEqualLonBoundsToDomain:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)
        self.domain_bounds = self.ds.lat_bnds

    def test_converts_region_bounds_to_domain_bounds_if_region_bounds_values_are_equal(
        self,
    ):
        result = self.ds.spatial._set_equal_lon_bounds_to_domain(
            domain_bounds=self.domain_bounds, region_bounds=np.array([50, 50])
        )
        expected = np.array([-90, 90])
        assert np.array_equal(result, expected)

    def test_returns_same_region_bounds_if_values_are_not_equal(self):
        result = self.ds.spatial._set_equal_lon_bounds_to_domain(
            domain_bounds=self.domain_bounds, region_bounds=np.array([0, 50])
        )
        expected = np.array([0, 50])
        assert np.array_equal(result, expected)


class TestScaleDimToRegion:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_scales_lat_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lat_bnds",
            data=np.array(
                [[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )
        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([-5, 5])
        )
        expected = xr.DataArray(
            name="lat_bnds",
            data=np.array([[-5.0, -5.0], [-5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)

    def test_scales_lon_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [359.0625, 360.9375],
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )
        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([190, 240])
        )
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [[240.0, 240.0], [190.0, 190.0], [190.0, 240.0], [240.0, 240.0]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )
        assert result.identical(expected)

    def test_scales_lon_bounds_when_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    # Does not apply to any conditional.
                    [359.0625, 360.9375],
                    # Grid cells stradling upper boundary.
                    [0.9375, 179.0625],
                    # Grid cells in between boundaries.
                    [179.0625, 357.1875],
                    # Grid cell straddling lower boundary.
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )
        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([357.5, 10.0])
        )
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    # Does not apply to any conditional.
                    [359.0625, 360.9375],
                    # Grid cells stradling upper boundary.
                    [0.9375, 10.0],
                    # Grid cells in between boundaries.
                    [10.0, 10.0],
                    # Grid cell straddling lower boundary.
                    [357.5, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )
        assert result.identical(expected)


class TestCombineWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)
        self.axis_weights = {
            "lat": xr.DataArray(
                name="lat_wts",
                data=np.array([1, 2, 3, 4]),
                coords={"lat": self.ds.lat},
                dims=["lat"],
            ),
            "lon": xr.DataArray(
                name="lon_wts",
                data=np.array([1, 2, 3, 4]),
                coords={"lon": self.ds.lon},
                dims=["lon"],
            ),
        }

    def test_weights_for_single_axes_is_the_same(self):
        result = self.ds.spatial._combine_weights(
            axis=["lat"], axis_weights=self.axis_weights
        )
        expected = self.axis_weights["lat"]

        assert result.identical(expected)

    def test_weights_for_multiple_axes_is_a_matrix_multiplication(self):
        result = self.ds.spatial._combine_weights(
            axis=["lat", "lon"], axis_weights=self.axis_weights
        )
        expected = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        assert result.identical(expected)


class TestAverager:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_weighted_avg_over_lat_axes(self):
        weights = xr.DataArray(
            name="lat_wts",
            data=np.array([1, 2, 3, 4]),
            coords={"lat": self.ds.lat},
            dims=["lat"],
        )
        result = self.ds.spatial._averager(self.ds.ts, axis=["lat"], weights=weights)
        expected = xr.DataArray(
            name="ts",
            data=np.ones((14, 4)),
            coords={"time": self.ds.time, "lon": self.ds.lon},
            dims=["time", "lon"],
        )

        assert result.identical(expected)

    def test_weighted_avg_over_lon_axes(self):
        weights = xr.DataArray(
            name="lon_wts",
            data=np.array([1, 2, 3, 4]),
            coords={"lon": self.ds.lon},
            dims=["lon"],
        )
        result = self.ds.spatial._averager(self.ds.ts, axis=["lon"], weights=weights)
        expected = xr.DataArray(
            name="ts",
            data=np.ones((14, 4)),
            coords={"time": self.ds.time, "lat": self.ds.lat},
            dims=["time", "lat"],
        )

        assert result.identical(expected)

    def test_weighted_avg_over_lat_and_lon_axes(self):
        weights = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )
        result = self.ds.spatial._averager(
            self.ds.ts, axis=["lat", "lon"], weights=weights
        )
        expected = xr.DataArray(
            name="ts", data=np.ones(14), coords={"time": self.ds.time}, dims=["time"]
        )

        assert result.identical(expected)
