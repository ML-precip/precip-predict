import numpy as np
import xarray as xr
import pandas as pd
from datetime import date

CH_CENTER = [46.818, 8.228]
CH_BOUNDING_BOX = [45.66, 47.87, 5.84, 10.98]
G = 9.80665

# Data extraction functions for ERA5


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


def extract_nearest_point_data(ds, lat, lon, level=0):
    """Return the time series data for the nearest grid point.

    Arguments:
        ds -- the dataset (xarray Dataset) to extract the data from
        lat -- the latitude coordinate of the point of interest
        lon -- the longitude coordinate of the point of interest
        level -- the desired vertical level

    Example:
    z = xr.open_mfdataset(DATADIR + '/ERA5/geopotential/*.nc', combine='by_coords')
    a = extract_nearest_point_data(z, CH_CENTER[0], CH_CENTER[1])
    """
    if 'level' in ds.dims:
        return ds.sel({'lat': lat, 'lon': lon, 'level': level}, method="nearest")

    return ds.sel({'lat': lat, 'lon': lon}, method="nearest")


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


def get_data_mean_over_box(ds, lats, lons, level=0):
    """Extract data from points within a bounding box and process the mean.

    Arguments:
    ds -- the dataset (xarray Dataset) to extract the data from
    lats -- the min/max latitude coordinates of the bounding box
    lons -- the min/max longitude coordinates of the bounding box
    level -- the desired vertical level
    """
    if len(lats) != 2:
        raise Exception('An array of length 2 is expected for the lats.')
    if len(lons) != 2:
        raise Exception('An array of length 2 is expected for the lons.')

    lat_start = min(lats)
    lat_end = max(lats)

    if (ds.lat[0] > ds.lat[1]):
        lat_start = max(lats)
        lat_end = min(lats)

    if 'level' in ds.dims:
        ds_box = ds.sel(
            lat=slice(lat_start, lat_end), lon=slice(min(lons), max(lons)), level=level
        )
    else:
        ds_box = ds.sel(
            lat=slice(lat_start, lat_end), lon=slice(min(lons), max(lons))
        )

    return ds_box.mean(['lat', 'lon'])


def get_data_mean_over_CH_box(ds, level=0):
    """Extract data over the bounding box of Switzerland and return the mean time series.

    Arguments:
    level -- the desired vertical level
    """
    return get_data_mean_over_box(ds, [CH_BOUNDING_BOX[0], CH_BOUNDING_BOX[1]],
                                  [CH_BOUNDING_BOX[2], CH_BOUNDING_BOX[3]], level)


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

def precip_exceedance_xarray(precip, qt):
    """Create exceedances of precipitation

    Arguments:
    precip -- xarray with precipitation 
    qt -- the desired quantile
    """
    qq = xr.DataArray(precip.tp).quantile(qt, dim='time') 
    out = xr.DataArray(precip.tp > qq)
    out = out*1
    # mark values
    # xr.DataArray(precip.tp).where(out==1)

    return out

def get_precipitation_data(path, start, end):
    """Read the precipitation time series and select data for the given period

    Arguments:
    path -- path to the csv file
    start -- start of the period to extract
    end -- end of the period to extract
    """
    precip = pd.read_csv(path)

    df_time = pd.to_datetime({
        'year': precip.year,
        'month': precip.month,
        'day': precip.day})
    precip.insert(0, "date", df_time, True)

    precip = precip[(precip.date >= start) & (precip.date <= end)]

    return precip


def convert_units(df):
    """Convert each column to the corresponding units"""
    for key, ts in df.iteritems():
        if key in ['T2MMEAN', 'T2M']:
            df[key] = ts - 273.15
        if key in ['Z1000', 'Z850', 'Z700', 'Z500', 'Z300', '1000', '850', '700', '500', '300']:
            df[key] = ts/G
        if key in ['MSL']:
            df[key] = ts/100

    return df


def remove_duplicate_date_column(df):
    """Remove duplicate date column in dataframes"""
    names = df.columns.values.tolist()

    if names.count('date') > 1:
        dates = df.date
        dates = dates.iloc[:, -1:]
        df = df.drop('date', axis=1)
        df = pd.concat([dates, df], axis=1)

    return df


def concat_dataframes(dfs):
    """Concatenate dataframes provided as a list"""
    length = len(dfs[0])
    start_date_ref = None
    end_date_ref = None

    # Check consistency between dataframes
    for df in dfs:
        if len(df) != length:
            raise Exception(
                'Dataframes to concatenate do not have the same length ({} vs {})'.format(len(df), length))
        if 'date' in df.columns:
            start_date = df.date.iloc[0]
            end_date = df.date.iloc[-1]
            if type(start_date) is pd.Timestamp:
                start_date = start_date.date()
                end_date = end_date.date()
            if start_date_ref == None:
                start_date_ref = start_date
                end_date_ref = end_date
            if start_date != start_date_ref:
                raise Exception(
                    'Dataframes to concatenate do not have the same starting date')
            if end_date != end_date_ref:
                raise Exception(
                    'Dataframes to concatenate do not have the same ending date')

    dfs_concat = pd.concat(dfs, axis=1)

    if 'date' in dfs_concat.columns:
        dfs_concat = remove_duplicate_date_column(dfs_concat)

    return dfs_concat


def read_csv_files(csv_files, start, end, rename_columns=False):
    """"Read CSV files according to the pattern csv_files
        Select the time period """
    dataframes = []  # a list to hold all the individual pandas DataFrames
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, parse_dates=['date'])
        df['date'] = df['date'].dt.date
        df = df[(df.date >= start_date) & (df.date <= end_date)]
        df = convert_units(df)
        dataframes.append(df)

    if rename_columns:
        for df_num, df in enumerate(dataframes):
            for column in df.columns:
                if column == 'date':
                    continue
                df.rename(columns={column: f'{column}_{df_num}'}, inplace=True)

    all_dfs = concat_dataframes(dataframes)

    all_dfs.date = pd.to_datetime(all_dfs.date, errors='coerce')

    return all_dfs


def prepare_prec_data_by_aggregated_regions(df, qt=0.95):
    """Prepare dataframe by aggregated regions

    Arguments:
    precip -- the precipitation dataframe
    qt -- the desired quantile
    """
    df = df[['date', 'reg_aggreg_1', 'reg_aggreg_2', 'reg_aggreg_3',
             'reg_aggreg_4', 'reg_aggreg_5', 'reg_tot']]
    df = df.rename(columns={'reg_aggreg_1': 'reg_1', 'reg_aggreg_2': 'reg_2',
                            'reg_aggreg_3': 'reg_3', 'reg_aggreg_4': 'reg_4',
                            'reg_aggreg_5': 'reg_5'})

    df_xtrm = precip_exceedance(df, qt)
    df_xtrm = df_xtrm.rename(columns={'reg_1': 'reg_1_xtr', 'reg_2': 'reg_2_xtr',
                                      'reg_3': 'reg_3_xtr', 'reg_4': 'reg_4_xtr',
                                      'reg_5': 'reg_5_xtr', 'reg_tot': 'reg_tot_xtr'})

    df = pd.concat([df, df_xtrm], axis=1)

    df = remove_duplicate_date_column(df)

    return df



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