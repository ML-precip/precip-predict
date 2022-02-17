import numpy as np
import xarray as xr


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