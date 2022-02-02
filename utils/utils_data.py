import numpy as np
import xarray as xr

G = 9.80665


def rename_dimensions_variables(ds):
    """Rename dimensions and attributes of the given dataset to homogenize data."""
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})

    return ds


def get_era5_data(files, start, end):
    """Extract ERA5 data for the given file(s) pattern/path."""
    print('Extracting data for the period {} - {}'.format(start, end))
    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = rename_dimensions_variables(ds)
    ds = ds.sel(
        time=slice(start, end)
    )

    if 'time_bnds' in ds.variables:
        ds = ds.drop('time_bnds')

    return ds


def extract_points_around(ds, lat, lon, step_lat, step_lon, nb_lat, nb_lon, levels=0):
    """Return the time series data for a grid point mesh around the provided coordinates.

    Arguments:
    ds -- the dataset (xarray Dataset) to extract the data from
    lat -- the latitude coordinate of the center of the mesh
    lon -- the longitude coordinate of the center of the mesh
    step_lat -- the step in latitude of the mesh
    step_lon -- the step in longitude of the mesh
    nb_lat -- the total number of grid points to extract for the latitude axis (the mesh will be centered)
    nb_lon -- the total number of grid points to extract for the longitude axis (the mesh will be centered)
    levels -- the desired vertical level(s)

    Example:
    z = xr.open_mfdataset(DATADIR + '/ERA5/geopotential/*.nc', combine='by_coords')
    a = extract_points_around(z, CH_CENTER[0], CH_CENTER[1], step_lat=1, step_lon=1, nb_lat=3, nb_lon=3)
    """
    lats = np.arange(lat - step_lat * (nb_lat - 1) / 2,
                     lat + step_lat * nb_lat / 2, step_lat)
    lons = np.arange(lon - step_lon * (nb_lon - 1) / 2,
                     lon + step_lon * nb_lon / 2, step_lon)

    if 'level' in ds.dims:
        data = ds.sel({'lat': lats, 'lon': lons,
                      'level': levels}, method='nearest')
    else:
        data = ds.sel({'lat': lats, 'lon': lons}, method='nearest')

    return data


def extract_points_around_CH(ds, step_lat, step_lon, nb_lat, nb_lon, levels=0):
    """Return the time series data for a grid point mesh around Switzerland.

    Arguments:
    ds -- the dataset (xarray Dataset) to extract the data from
    step_lat -- the step in latitude of the mesh
    step_lon -- the step in longitude of the mesh
    nb_lat -- the total number of grid points to extract for the latitude axis (the mesh will be centered)
    nb_lon -- the total number of grid points to extract for the longitude axis (the mesh will be centered)
    levels -- the desired vertical level(s)
    """
    return extract_points_around(ds, CH_CENTER[0], CH_CENTER[1], step_lat=step_lat, step_lon=step_lon,
                                 nb_lat=nb_lat, nb_lon=nb_lon, levels=levels)


def precip_exceedance(precip, qt=0.95):
    """Create exceedances of precipitation

    Arguments:
    precip -- the precipitation dataframe
    qt -- the desired quantile
    """
    precip_qt = precip.copy()

    for key, ts in precip.iteritems():
        if key in ['date', 'year', 'month', 'day']:
            continue
        precip_qt[key] = ts > ts.quantile(qt)

    return precip_qt


def precip_exceedance_xarray(precip, qt=0.95):
    """Create exceedances of precipitation

    Arguments:
    precip -- xarray with precipitation 
    qt -- the desired quantile
    """
    qq = xr.DataArray(precip.tp).quantile(qt, dim='time') 
    out = xr.DataArray(precip.tp > qq)
    out = out*1

    return out


def get_Y_sets(dat, YY_TRAIN, YY_VAL, YY_TEST):
    """ Prepare the targe Y for train, validation and test """
    # Prepare the target Y
    # Split the prec into the same 
    Y_train = dat.sel(time=slice('{}-01-01'.format(YY_TRAIN[0]),
                             '{}-12-31'.format(YY_VAL)))
    Y_val = dat.sel(time=slice('{}-01-01'.format(YY_TRAIN[1]),
                             '{}-12-31'.format(YY_TRAIN[1])))
    Y_test = dat.sel(time=slice('{}-01-01'.format(YY_TEST[0]),
                            '{}-12-31'.format(YY_TEST[1])))
    Y_train_input = np.array(Y_train)
    Y_val_input = np.array(Y_val)
    Y_test_input = np.array(Y_test)

    return Y_train_input, Y_val_input, Y_test_input