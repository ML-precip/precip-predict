import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import dask
import datetime
from dotenv import dotenv_values

def rename_dimensions_variables(ds):
    """Rename dimensions and attributes of the given dataset to homogenize data."""
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})

    return ds


def temporal_slice(ds, start, end):
    """Slice along the temporal dimension."""
    ds = ds.sel(time=slice(start, end))

    if 'time_bnds' in ds.variables:
        ds = ds.drop('time_bnds')

    return ds


def spatial_slice(ds, lon_bnds, lat_bnds):
    """Slice along the spatial dimension."""
    if lon_bnds != None:
        ds = ds.sel(lon=slice(min(lon_bnds), max(lon_bnds)))

    if lat_bnds != None:
        if ds.lat[0].values < ds.lat[1].values:
            ds = ds.sel(lat=slice(min(lat_bnds), max(lat_bnds)))
        else:
            ds = ds.sel(lat=slice(max(lat_bnds), min(lat_bnds)))

    return ds


def get_nc_data(files, start, end, lon_bnds=None, lat_bnds=None):
    """Extract netCDF data for the given file(s) pattern/path."""
    print('Extracting data for the period {} - {}'.format(start, end))
    ds = xr.open_mfdataset(files, combine='by_coords')
    ds = rename_dimensions_variables(ds)
    ds = temporal_slice(ds, start, end)
    ds = spatial_slice(ds, lon_bnds, lat_bnds)

    return ds


def get_era5_data(files, start, end, lon_bnds=None, lat_bnds=None):
    """Extract ERA5 data for the given file(s) pattern/path."""
    
    return get_nc_data(files, start, end, lon_bnds, lat_bnds)


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
    qq = xr.DataArray(precip).quantile(qt, dim='time') 
    out = xr.DataArray(precip > qq)
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




def load_data(i_vars, i_paths, G, PATH_ERA5, DATE_START, DATE_END, LONS, LATS, LEVELS):
    """Load the data
       Args: 
       Var: variables
       PATH_ERA5: path to the era5 datasets
       DATE_START: starting date
       DATE_END: end date
       LONS: longitudes
       LATS: latitudes"""

    
    l_vars = []
    for iv in range(0,len(i_vars)):
        
       # if i_vars[iv] == 't2m':
       #     vv = get_era5_data(PATH_ERA5 + i_paths[iv] +'Grid1_Daymean_era5_T2M_EU_19790101-20211231.nc', DATE_START, DATE_END, LONS, LATS)
       # else:
        vv = get_era5_data(PATH_ERA5 + i_paths[iv] +'*nc', DATE_START, DATE_END, LONS, LATS)
         
        if i_vars[iv] == 'z':
            vv = vv.sel(level=LEVELS)
            vv.z.values = vv.z.values/G
        elif i_vars[iv] == 'rh':
            vv = vv.sel(level=LEVELS)
            
        vv['time'] = pd.DatetimeIndex(vv.time.dt.date)
    
        l_vars.append(vv)

    
    
    return l_vars