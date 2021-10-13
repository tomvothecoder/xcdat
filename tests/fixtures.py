"""This module stores reusable test fixtures."""
from datetime import datetime

import numpy as np
import xarray as xr

# If the fixture is an xarray object, make sure to use .copy() to create a
# shallow copy of the object. Otherwise, you might run into unintentional
# side-effects caused by reference assignment.
# https://xarray.pydata.org/en/stable/generated/xarray.DataArray.copy.html

# NOTE:
# - Non-CF time includes "units" attr
# - Coordinates with bounds includes "bounds" attr and vice versa

# TIME
# ====
time_cf = xr.DataArray(
    data=[
        datetime(2000, 1, 1),
        datetime(2000, 2, 1),
        datetime(2000, 3, 1),
        datetime(2000, 4, 1),
        datetime(2000, 5, 1),
        datetime(2000, 6, 1),
        datetime(2000, 7, 1),
        datetime(2000, 8, 1),
        datetime(2000, 9, 1),
        datetime(2000, 10, 1),
        datetime(2000, 11, 1),
        datetime(2000, 12, 1),
        datetime(2001, 1, 1),
        datetime(2001, 2, 1),
    ],
    dims=["time"],
    attrs={
        "long_name": "time",
        "standard_name": "time",
        "axis": "T",
    },
)
time_non_cf = xr.DataArray(
    data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    dims=["time"],
    attrs={
        "long_name": "time",
        "standard_name": "time",
        "axis": "T",
    },
)

time_bnds = xr.DataArray(
    name="time_bnds",
    data=[
        [datetime(1999, 12, 16, 12), datetime(2000, 1, 16, 12)],
        [datetime(2000, 1, 16, 12), datetime(2000, 2, 15, 12)],
        [datetime(2000, 2, 15, 12), datetime(2000, 3, 16, 12)],
        [datetime(2000, 3, 16, 12), datetime(2000, 4, 16)],
        [datetime(2000, 4, 16), datetime(2000, 5, 16, 12)],
        [datetime(2000, 5, 16, 12), datetime(2000, 6, 16)],
        [datetime(2000, 6, 16), datetime(2000, 7, 16, 12)],
        [datetime(2000, 7, 16, 12), datetime(2000, 8, 16, 12)],
        [datetime(2000, 8, 16, 12), datetime(2000, 9, 16)],
        [datetime(2000, 9, 16), datetime(2000, 10, 16, 12)],
        [datetime(2000, 10, 16, 12), datetime(2000, 11, 16)],
        [datetime(2000, 11, 16), datetime(2000, 12, 16, 12)],
        [datetime(2000, 12, 16, 12), datetime(2001, 1, 16, 12)],
        [datetime(2001, 1, 16, 12), datetime(2001, 2, 16, 12)],
    ],
    coords={"time": time_cf},
    dims=["time", "bnds"],
    attrs={
        "is_generated": "True",
    },
)
time_bnds_non_cf = xr.DataArray(
    name="time_bnds",
    data=[
        [datetime(1999, 12, 16, 12), datetime(2000, 1, 16, 12)],
        [datetime(2000, 1, 16, 12), datetime(2000, 2, 15, 12)],
        [datetime(2000, 2, 15, 12), datetime(2000, 3, 16, 12)],
        [datetime(2000, 3, 16, 12), datetime(2000, 4, 16)],
        [datetime(2000, 4, 16), datetime(2000, 5, 16, 12)],
        [datetime(2000, 5, 16, 12), datetime(2000, 6, 16)],
        [datetime(2000, 6, 16), datetime(2000, 7, 16, 12)],
        [datetime(2000, 7, 16, 12), datetime(2000, 8, 16, 12)],
        [datetime(2000, 8, 16, 12), datetime(2000, 9, 16)],
        [datetime(2000, 9, 16), datetime(2000, 10, 16, 12)],
        [datetime(2000, 10, 16, 12), datetime(2000, 11, 16)],
        [datetime(2000, 11, 16), datetime(2000, 12, 16)],
        [datetime(2001, 12, 16), datetime(2001, 1, 16)],
        [datetime(2001, 1, 16), datetime(2001, 2, 15)],
    ],
    coords={"time": time_non_cf},
    dims=["time", "bnds"],
    attrs={"is_generated": "True"},
)

time_cf_hourly = xr.DataArray(
    data=[
        datetime(2000, 1, 1, 12),
        datetime(2000, 2, 1, 6),
        datetime(2000, 3, 1, 12),
        datetime(2000, 4, 1, 6),
        datetime(2000, 5, 1, 12),
        datetime(2000, 6, 1, 6),
        datetime(2000, 7, 1, 12),
        datetime(2000, 8, 1, 6),
        datetime(2000, 9, 1, 12),
        datetime(2000, 10, 1, 6),
        datetime(2000, 11, 1, 12),
        datetime(2000, 12, 1, 6),
        datetime(2001, 1, 1, 12),
        datetime(2001, 2, 1, 6),
    ],
    dims=["time"],
    attrs={
        "long_name": "time",
        "standard_name": "time",
        "axis": "T",
    },
)
time_bnds_hourly = xr.DataArray(
    name="time_bnds",
    data=[
        [datetime(1999, 12, 17, 3), datetime(2000, 1, 16, 21)],
        [datetime(2000, 1, 16, 21), datetime(2000, 2, 15, 21)],
        [datetime(2000, 2, 15, 21), datetime(2000, 3, 16, 21)],
        [datetime(2000, 3, 16, 21), datetime(2000, 4, 16, 9)],
        [datetime(2000, 4, 16, 9), datetime(2000, 5, 16, 21)],
        [datetime(2000, 5, 16, 21), datetime(2000, 6, 16, 9)],
        [datetime(2000, 6, 16, 9), datetime(2000, 7, 16, 21)],
        [datetime(2000, 7, 16, 21), datetime(2000, 8, 16, 21)],
        [datetime(2000, 8, 16, 21), datetime(2000, 9, 16, 9)],
        [datetime(2000, 9, 16, 9), datetime(2000, 10, 16, 21)],
        [datetime(2000, 10, 16, 21), datetime(2000, 11, 16, 9)],
        [datetime(2000, 11, 16, 9), datetime(2000, 12, 16, 21)],
        [datetime(2000, 12, 16, 21), datetime(2001, 1, 16, 21)],
        [datetime(2001, 1, 16, 21), datetime(2001, 2, 16, 15)],
    ],
    coords={"time": time_cf_hourly},
    dims=["time", "bnds"],
    attrs={
        "is_generated": "True",
    },
)

# LATITUDE
# ========
lat = xr.DataArray(
    data=np.array([-90, -88.75, 88.75, 90]),
    dims=["lat"],
    attrs={"units": "degrees_north", "axis": "Y"},
)
lat_bnds = xr.DataArray(
    name="lat_bnds",
    data=np.array([[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]),
    coords={"lat": lat},
    dims=["lat", "bnds"],
    attrs={"is_generated": "True"},
)

# LONGITUDE
# =========
lon = xr.DataArray(
    data=np.array([0, 1.875, 356.25, 358.125]),
    dims=["lon"],
    attrs={"units": "degrees_east", "axis": "X"},
)
lon_bnds = xr.DataArray(
    name="lon_bnds",
    data=np.array(
        [
            [-0.9375, 0.9375],
            [0.9375, 179.0625],
            [179.0625, 357.1875],
            [357.1875, 359.0625],
        ]
    ),
    coords={"lon": lon},
    dims=["lon", "bnds"],
    attrs={"is_generated": "True"},
)

# VARIABLES
# =========
ts_cf = xr.DataArray(
    name="ts",
    data=np.ones((14, 4, 4)),
    coords={"time": time_cf, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)

ts_non_cf = xr.DataArray(
    name="ts",
    data=np.ones((14, 4, 4)),
    coords={"time": time_non_cf, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)


def generate_dataset(cf_compliant: bool, has_bounds: bool) -> xr.Dataset:
    """Generates a dataset using coordinate and data variable fixtures.

    Parameters
    ----------
    cf_compliant : bool, optional
        CF compliant time units.
    has_bounds : bool, optional
        Include bounds for coordinates. This also adds the "bounds" attribute
        to existing coordinates to link them to their respective bounds.

    Returns
    -------
    xr.Dataset
        Test dataset.
    """
    if has_bounds:
        ds = xr.Dataset(
            data_vars={
                "ts": ts_cf.copy(),
                "lat_bnds": lat_bnds.copy(),
                "lon_bnds": lon_bnds.copy(),
            },
            coords={"lat": lat.copy(), "lon": lon.copy()},
        )

        if cf_compliant:
            ds = ds.assign({"time_bnds": time_bnds.copy()})
            ds = ds.assign_coords({"time": time_cf.copy()})
        elif not cf_compliant:
            ds = ds.assign({"time_bnds": time_bnds_non_cf.copy()})
            ds = ds.assign_coords({"time": time_non_cf.copy()})
            ds["time"] = ds.time.assign_attrs(units="months since 2000-01-01")

        # If the "bounds" attribute is included in an existing DataArray and
        # added to a new Dataset, it will get dropped. Therefore, it needs to be
        # assigned to the DataArrays after they are added to Dataset.
        ds["lat"] = ds.lat.assign_attrs(bounds="lat_bnds")
        ds["lon"] = ds.lon.assign_attrs(bounds="lon_bnds")
        ds["time"] = ds.time.assign_attrs(bounds="time_bnds")

    elif not has_bounds:
        ds = xr.Dataset(
            data_vars={"ts": ts_cf.copy()},
            coords={"lat": lat.copy(), "lon": lon.copy()},
        )

        if cf_compliant:
            ds = ds.assign_coords({"time": time_cf.copy()})
        elif not cf_compliant:
            ds = ds.assign_coords({"time": time_non_cf.copy()})
            ds["time"] = ds.time.assign_attrs(units="months since 2000-01-01")

    return ds
