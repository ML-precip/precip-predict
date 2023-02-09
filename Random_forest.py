### Random Forest analysis
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= '0.20'

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.utils import class_weight

# Common imports
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask
import datetime
import math
import pickle
import pathlib
import hashlib
from pathlib import Path
#dask.config.set({'array.slicing.split_large_chunks': False})

# To make this notebook's output stable across runs
np.random.seed(42)

from dotenv import dotenv_values

# Custom utils
from utils.utils_data import *
from utils.utils_plot import *
from utils.utils_RF import *
from utils.utils_ml import *


config = dotenv_values(".env")

# Paths
PATH_ERA5 = config['PATH_ERA5']
PATH_EOBS = config['PATH_EOBS']

# Some constants
G = 9.80665

###########
# Options
##########
PRECIP_DATA = 'ERA5-low' # Options: ERA5-hi, ERA5-low, E-OBS
DATE_START = '1979-01-01'
DATE_END = '2021-12-31'
YY_TRAIN = [1979, 2015]
YY_TEST = [2016, 2021]
LEVELS = [300, 500, 700, 850, 925, 1000]
LONS_INPUT = [-25, 30]
LATS_INPUT = [30, 75]
LONS_PREC = [-25, 30]
LATS_PREC = [30, 75]
BATCH_SIZE = 64
PRECIP_XTRM = 0.95 # Percentile (threshold) for the extremes
CREATE_MASK_EOBS = False # This option is only true when using E-OBS
RF_MAX_DEPTH =[12,14,16] #[3,4,6,8,10]



# Load precipitation
if PRECIP_DATA == 'ERA5-hi':
    pr = get_nc_data(PATH_ERA5 + '/precipitation/orig_grid/daily/*nc', DATE_START, DATE_END, LONS_PREC, LATS_PREC)
    pr = pr.tp
elif PRECIP_DATA == 'ERA5-low':
    pr = get_nc_data(PATH_ERA5 + '/precipitation/day_grid1/*nc', DATE_START, DATE_END, LONS_PREC, LATS_PREC)
    pr = pr.tp
elif PRECIP_DATA in ['E-OBS', 'EOBS']:
    pr = get_nc_data(PATH_EOBS + '/eobs_1deg_v26.0e.nc', DATE_START, DATE_END, LONS_PREC, LATS_PREC)
    pr = pr.rr
    CREATE_MASK_EOBS = True
else:
    raise ValueError('Precipitation data not well defined')
pr['time'] = pd.DatetimeIndex(pr.time.dt.date)

# Invert lat axis if needed
if pr.lat[0].values < pr.lat[1].values:
    pr = pr.reindex(lat=list(reversed(pr.lat)))


# Create mask
mask = None
if CREATE_MASK_EOBS:
    if PRECIP_DATA in ['E-OBS', 'EOBS']:
        peobs = pr
    else:
        peobs = get_nc_data(PATH_EOBS + '/eobs_1deg_v26.0e.nc', DATE_START, DATE_END, LONS_PREC, LATS_PREC)
        peobs = peobs.rr
        if peobs.lat[0].values < peobs.lat[1].values:
            peobs = peobs.reindex(lat=list(reversed(peobs.lat)))
    mask = np.isnan(peobs[0,:,:])
    mask = np.invert(mask)
    mask.plot()
    mask = mask.to_numpy()


# Compute the extreme exceedence
qq = xr.DataArray(pr).chunk(dict(time=-1)).quantile(PRECIP_XTRM, dim='time')
pr_xtrm = xr.DataArray(pr > qq)
pr_xtrm = pr_xtrm*1 # Transform to number

lats_y = pr.lat.to_numpy()
lons_y = pr.lon.to_numpy()


## Input data: meteorological fields

with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    # Load geopotential height
    z = get_era5_data(PATH_ERA5 + '/geopotential/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
    z['time'] = pd.DatetimeIndex(z.time.dt.date)
    z = z.sel(level=LEVELS)

    # Get Z in geopotential height (m)
    z.z.values = z.z.values/G

    # Load temperature
    t = get_era5_data(PATH_ERA5 + '/temperature/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
    t['time'] = pd.DatetimeIndex(t.time.dt.date)

    # Load relative humidity
    rh = get_era5_data(PATH_ERA5 + '/relative_humidity/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
    rh['time'] = pd.DatetimeIndex(rh.time.dt.date)
    rh = rh.sel(level=LEVELS)

    # Load total column water
    tcw = get_era5_data(PATH_ERA5 + '/total_column_water/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
    tcw['time'] = pd.DatetimeIndex(tcw.time.dt.date)

    # Load wind components
    u = get_era5_data(PATH_ERA5 + '/U_wind/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
    u['time'] = pd.DatetimeIndex(u.time.dt.date)
    v = get_era5_data(PATH_ERA5 + '/V_wind/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
    v['time'] = pd.DatetimeIndex(v.time.dt.date)

# Merge arrays
with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    X = xr.merge([z, t, rh, tcw, u, v])

# Invert lat axis if needed
if X.lat[0].values < X.lat[1].values:
    X = X.reindex(lat=list(reversed(X.lat)))
    
### Split data and transform

X_train_full = X.sel(time=slice(f'{YY_TRAIN[0]}-01-01', f'{YY_TRAIN[1]}-12-31'))
X_test = X.sel(time=slice(f'{YY_TEST[0]}-01-01', f'{YY_TEST[1]}-12-31'))

pr_train_full = pr.sel(time=slice(f'{YY_TRAIN[0]}-01-01', f'{YY_TRAIN[1]}-12-31'))
pr_test = pr.sel(time=slice(f'{YY_TEST[0]}-01-01', f'{YY_TEST[1]}-12-31'))

pr_xtrm_train_full = pr_xtrm.sel(time=slice(f'{YY_TRAIN[0]}-01-01', f'{YY_TRAIN[1]}-12-31'))
pr_xtrm_test = pr_xtrm.sel(time=slice(f'{YY_TEST[0]}-01-01', f'{YY_TEST[1]}-12-31'))

# Create a data generator
dic = {'z': LEVELS,
   't': LEVELS,
   'r': LEVELS,
   'tcwv': None,
   'u': LEVELS,
   'v': LEVELS}



YY_VALID = 2005
# we might not need to split into valid
dg_train = DataGeneratorWithExtremes(X_train_full.sel(time=slice(f'{YY_TRAIN[0]}', f'{YY_VALID}')),
                                     pr_train_full.sel(time=slice(f'{YY_TRAIN[0]}', f'{YY_VALID}')),
                                     pr_xtrm_train_full.sel(time=slice(f'{YY_TRAIN[0]}', f'{YY_VALID}')),
                                     dic, batch_size=BATCH_SIZE, load=True)
dg_valid = DataGeneratorWithExtremes(X_train_full.sel(time=slice(f'{YY_VALID+1}', f'{YY_TRAIN[1]}')),
                                     pr_train_full.sel(time=slice(f'{YY_VALID+1}', f'{YY_TRAIN[1]}')),
                                     pr_xtrm_train_full.sel(time=slice(f'{YY_VALID+1}', f'{YY_TRAIN[1]}')),
                                     dic, mean=dg_train.mean, std=dg_train.std,
                                     batch_size=BATCH_SIZE, load=True)
dg_test = DataGeneratorWithExtremes(X_test, pr_test, pr_xtrm_test, dic,
                                    mean=dg_train.mean, std=dg_train.std,
                                    batch_size=BATCH_SIZE, load=True, shuffle=False)


i_shape = dg_train.X.shape
o_shape = dg_train.y.shape

print(f'X shape: {i_shape}')
print(f'y shape: {o_shape}')

chunk = {'lon':5, 'lat':5}

print('starting RF')
## Train a random forest classifier

X_train_dask = dg_train.X.chunk(chunk)
y_train_dask = dg_train.y_xtrm.chunk(chunk)

# target for the regression
yreg_train_dask = dg_train.y.chunk(chunk)

for i in RF_MAX_DEPTH:
    
    print('fit RF for depth',i)
    # models file
    clf_file = f'tmp/RF/trained_classifiers_RF_{PRECIP_DATA}_{PRECIP_XTRM}_{i}.pkl'
    
    if os.path.isfile(clf_file):
        print('open models')
        #open models
        with open(clf_file, 'rb') as f:
            crfs = pickle.load(f)
    else:

        crfs = xr.apply_ufunc(
            train_rf_classifier_model,
            X_train_dask, y_train_dask, i,
            vectorize=True,
            dask = 'parallelized',
            input_core_dims=[['time', 'level'], ['time'], []],  # reduce along these dimensions
            output_dtypes=[object]
        ).compute()
        # save the models
        with open(clf_file, 'wb') as output:
                pickle.dump(crfs, output, pickle.HIGHEST_PROTOCOL)
                
                
                
    # Regressor part
    # models file
    rfs_file = f'tmp/RF/trained_regressors_RF_{PRECIP_DATA}_{i}.pkl'
    

    if os.path.isfile(rfs_file):
        #print('open models')
        #open models
        with open(rfs_file, 'rb') as f:
            rfs = pickle.load(f)
    else:

        rfs = xr.apply_ufunc(
            train_rf_regress_model,
            X_train_dask, y_train_dask, i,
            vectorize=True,
            dask = 'parallelized',
            input_core_dims=[['time', 'level'], ['time'], []],  # reduce along these dimensions
            output_dtypes=[object]
        ).compute()
        # save the models
        with open(rfs_file, 'wb') as output:
                pickle.dump(rfs, output, pickle.HIGHEST_PROTOCOL)


