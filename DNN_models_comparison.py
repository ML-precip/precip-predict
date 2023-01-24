# Script to run the main models used to predict precipitation and the extremes
# This version uses Keras (instead tensorflow) as in the *ipynb version.

import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= '0.20'

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
dask.config.set({'array.slicing.split_large_chunks': False})

# To make this notebook's output stable across runs
np.random.seed(42)

# Dotenv
from dotenv import dotenv_values

# Custom utils
from utils.utils_data import *
from utils.utils_ml import *
from utils.utils_resnet import *
from utils.utils_plot import *
from utils.DNN_models import *

config = dotenv_values(".env")

# Paths
PATH_ERA5 = config['PATH_ERA5']
PATH_EOBS = config['PATH_EOBS']

# Some constants
G = 9.80665

# Options
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
PRECIP_DATA = 'E-OBS' # Options: ERA5-hi, ERA5-low, E-OBS
PRECIP_XTRM = 0.95 # Percentile (threshold) for the extremes
USE_3D_ONLY = False


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
    #pr = pr.fillna(0) # Over the seas. Not optimal...
else:
    raise ValueError('Precipitation data not well defined')

    # Add a dimension to be used as channel in the DNN
pr = pr.expand_dims('level', -1)


# Compute the extreme exceedence
qq = xr.DataArray(pr).quantile(PRECIP_XTRM, dim='time')
pr_xtrm = xr.DataArray(pr > qq)
pr_xtrm = pr_xtrm*1 # Transform to number

# Extract coordinates for precip
lats_y = pr.lat.to_numpy()
lons_y = pr.lon.to_numpy()

## Input data: meteorological fields
# Load geopotential height
print('loading predictors')
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
if not USE_3D_ONLY:
    tcw = get_era5_data(PATH_ERA5 + '/total_column_water/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
    tcw['time'] = pd.DatetimeIndex(tcw.time.dt.date)

# Load wind components
u = get_era5_data(PATH_ERA5 + '/U_wind/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
u['time'] = pd.DatetimeIndex(u.time.dt.date)
v = get_era5_data(PATH_ERA5 + '/V_wind/day_grid1/*.nc', DATE_START, DATE_END, LONS_INPUT, LATS_INPUT)
v['time'] = pd.DatetimeIndex(v.time.dt.date)

# Checking dimensions
print('dimension of pr:', pr.dims)
print('dimension of z', z.dims)
print('dimension of t:', t.dims)
print('dimension of rh:', rh.dims)
if not USE_3D_ONLY:
    print('dimension of tcw:', tcw.dims)
print('dimension of u:', u.dims)
print('dimension of v:', v.dims)

# Merge arrays
if USE_3D_ONLY:
    X = xr.merge([z, t, rh, u, v])
else:
    X = xr.merge([z, t, rh, tcw, u, v])

# Invert lat axis if needed
if X.lat[0].values < X.lat[1].values:
    X = X.reindex(lat=list(reversed(X.lat)))
    
# Get axes
lats_x = X.lat
lons_x = X.lon

# Split into training and test
X_train_full = X.sel(time=slice(f'{YY_TRAIN[0]}-01-01', f'{YY_TRAIN[1]}-12-31'))
X_test = X.sel(time=slice(f'{YY_TEST[0]}-01-01', f'{YY_TEST[1]}-12-31'))

pr_train_full = pr.sel(time=slice(f'{YY_TRAIN[0]}-01-01', f'{YY_TRAIN[1]}-12-31'))
pr_test = pr.sel(time=slice(f'{YY_TEST[0]}-01-01', f'{YY_TEST[1]}-12-31'))

pr_xtrm_train_full = pr_xtrm.sel(time=slice(f'{YY_TRAIN[0]}-01-01', f'{YY_TRAIN[1]}-12-31'))
pr_xtrm_test = pr_xtrm.sel(time=slice(f'{YY_TEST[0]}-01-01', f'{YY_TEST[1]}-12-31'))

# Create a data generator
if USE_3D_ONLY:
    dic = {'z': LEVELS,
       't': LEVELS,
       'r': LEVELS,
       'u': LEVELS,
       'v': LEVELS}
else:
    dic = {'z': LEVELS,
       't': LEVELS,
       'r': LEVELS,
       'tcwv': None,
       'u': LEVELS,
       'v': LEVELS}



YY_VALID = 2005

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


i_shape = dg_train.X.shape[1:]
o_shape = dg_train.y.shape[1:]

print(f'X shape: {i_shape}')
print(f'y shape: {o_shape}')


mask = None
if PRECIP_DATA in ['E-OBS', 'EOBS']:
    mask = np.isnan(qq[:,:])
    mask = np.invert(mask)
    mask.plot()
    mask = mask.to_numpy()

# Computing the necessary output scaling.
dlons_x = float(lons_x[1] - lons_x[0])
dlats_x = float(lats_x[0] - lats_x[1])
dlons_y = float(lons_y[1] - lons_y[0])
dlats_y = float(lats_y[0] - lats_y[1])

output_scaling = int(dlons_x / dlons_y)
output_crop = None


def initiate_optimizer(lr_method, lr=.0004, init_lr=0.01, max_lr=0.01):
    if lr_method == 'Cyclical':
        # Cyclical learning rate
        steps_per_epoch = dg_train.n_samples // BATCH_SIZE
        clr = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=init_lr,
            maximal_learning_rate=max_lr,
            scale_fn=lambda x: 1/(2.**(x-1)),
            step_size=2 * steps_per_epoch)
        optimizer = tf.keras.optimizers.Adam(clr)
    elif lr_method == 'CosineDecay':
        decay_steps = EPOCHS * (dg_train.n_samples / BATCH_SIZE)
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            init_lr, decay_steps)
        optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)
    elif lr_method == 'Constant':
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    else:
        raise ValueError('learning rate schedule not well defined.')
        
    return optimizer


# Compute weights for the weighted binary crossentropy
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(pr_xtrm.values),
    y=pr_xtrm.values.flatten()
)

print('Weights for the weighted binary crossentropy:')
print(f'Classes: {np.unique(pr_xtrm.values)}, weights: {weights}')

# Create loss function for the extremes
xtrm_loss = weighted_binary_cross_entropy(
    weights={0: weights[0].astype('float32'), 1: weights[1].astype('float32')})



class MeanSquaredErrorNans(tf.keras.losses.Loss):
    def __init__(self, name="mse_nans", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        nb_values = tf.where(tf.math.is_nan(y_true),
                         tf.zeros_like(y_true),
                         tf.ones_like(y_true))
        nb_values = tf.reduce_sum(nb_values)
        y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)
        loss = tf.square(tf.subtract(y_pred, y_true))
        loss_sum = tf.reduce_sum(loss)
        return loss_sum / nb_values



# Define hyperparameters
EPOCHS = 200
LR_METHOD = 'Constant'  # Cyclical, CosineDecay, Constant
    
# Early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                            restore_best_weights=True)
                                            
# Default model options
opt_model = {'latent_dim': 128,
             'dropout_rate': 0.2}

# Default training options
opt_training = {'epochs': EPOCHS,
                'callbacks': [callback]}

# Default optimizer options
opt_optimizer = {'lr_method': 'Constant',
                 'lr': 0.0004,
                 'init_lr': 0.01}


models_report = {
          'Dav-orig': {'model': 'Davenport-2021', 'run': True,
                       'opt_model': {'latent_dim': 16},
                       'opt_optimizer': {'lr_method': 'Constant', 'lr': 4e-4}}, # original
          'Dav-64': {'model': 'Davenport-2021', 'run': True,
                     'opt_model': {'latent_dim': 64},
                     'opt_optimizer': {'lr_method': 'Constant', 'lr': 4e-4}},
          'Pan-orig': {'model': 'Pan-2019', 'run': True,
                       'opt_model': {'latent_dim': 60},
                       'opt_optimizer': {'lr_method': 'Constant', 'lr': 1e-4}},
          'CNN-2l': {'model': 'CNN-2L', 'run': True,
                     'opt_model': {'latent_dim': 64},
                     'opt_optimizer': {'lr_method': 'Constant', 'lr': 1e-3}}, # higher lr
          'UNET': {'model': 'Unet', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop, 'unet_depth': 4, 'use_upsample': False},
                   'opt_optimizer': {'lr_method': 'Constant', 'lr': 4e-4}}
         }


models_unets = {
          'UNET1': {'model': 'Unet', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop, 'unet_depth': 1, 'use_upsample': False},
                   'opt_optimizer': {'lr_method': 'Constant'}},
          'UNET2': {'model': 'Unet', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop, 'unet_depth': 2, 'use_upsample': False},
                   'opt_optimizer': {'lr_method': 'Constant'}},
          'UNET3': {'model': 'Unet', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop, 'unet_depth': 3, 'use_upsample': False},
                   'opt_optimizer': {'lr_method': 'Constant'}},
          'UNET4': {'model': 'Unet', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop, 'unet_depth': 4, 'use_upsample': False},
                   'opt_optimizer': {'lr_method': 'Constant'}},
         }


models_ranet = {
          'RaNet': {'model': 'RaNet', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop},
                   'opt_optimizer': {'lr_method': 'Constant'}},
         }



model_unet_simple = {
          'UNET2': {'model': 'Unet', 'run': True,
                   'opt_model': {'output_scaling': output_scaling, 'output_crop': output_crop, 'unet_depth': 2, 'use_upsample': False},
                   'opt_optimizer': {'lr_method': 'Constant'}},
         }



models = models_unets

train_for_prec = True
train_for_xtrm = True
history_log_level = 1

#loss_regression = 'mse'
loss_regression = 'mse_nans'

models_prec = []
models_xtrm = []

if train_for_prec:
        
    for m_id in models:
        # Clear session and set tf seed
        keras.backend.clear_session()
        tf.random.set_seed(42)
        
        if not models[m_id]['run']:
            continue

        # Extract model name and options
        model = models[m_id]['model']
        opt_model_i = models[m_id]['opt_model']
        opt_optimizer_i = models[m_id]['opt_optimizer']
        opt_model_new = opt_model.copy()
        opt_model_new.update(opt_model_i)
        opt_optimizer_new = opt_optimizer.copy()
        opt_optimizer_new.update(opt_optimizer_i)
      
        # Switch to precipitation values
        dg_train.for_extremes(False)
        dg_valid.for_extremes(False)
        dg_test.for_extremes(False)
        
        optimizer = initiate_optimizer(**opt_optimizer_new)

        # Load if previously saved
        tag = pickle.dumps(opt_model_new) + pickle.dumps(opt_optimizer_new) + pickle.dumps(i_shape + o_shape) + pickle.dumps(PRECIP_XTRM)
        hashed_name = f'prec_{m_id}_{hashlib.md5(tag).hexdigest()}'
        tmp_file = pathlib.Path(f'tmp/{hashed_name}')
        
        if tmp_file.exists():
            m = keras.models.load_model(tmp_file, custom_objects = {"MeanSquaredErrorNans": MeanSquaredErrorNans})
            
        else:
            # Create the model and compile
            m = DeepFactory_Keras(model, i_shape, o_shape, for_extremes=False, **opt_model_new)
            # Warning: When using regularizers, the loss function is the entire loss, ie (loss metrics) + (regularization term)!
            # But the loss displayed as part of the metrics, is only the loss metric. The regularization term is not added there. -> can be different!!
            loss_fct = 'mse'
            if loss_regression == 'mse_nans':
                loss_fct = MeanSquaredErrorNans()
            m.compile(
                loss=loss_fct, 
                metrics=[loss_fct], 
                optimizer=optimizer
            )
            print(f'Number of parameters: {m.model.count_params()}')

            # Train
            hist = m.fit(dg_train, validation_data=dg_valid, verbose=history_log_level, **opt_training)
        
            # Saving the model
            print('Saving weights')
            m.save.weights(f'tmp/keras/{model}.h5')
        
        
if train_for_xtrm:

    for m_id in models:
        # Clear session and set tf seed
        keras.backend.clear_session()
        tf.random.set_seed(42)

        if not models[m_id]['run']:
            continue
        
        # Extract model name and options
        model = models[m_id]['model']
        opt_model_i = models[m_id]['opt_model']
        opt_optimizer_i = models[m_id]['opt_optimizer']
        opt_model_new = opt_model.copy()
        opt_model_new.update(opt_model_i)
        opt_optimizer_new = opt_optimizer.copy()
        opt_optimizer_new.update(opt_optimizer_i)
        print(f'Running: {m_id} - {model} - {opt_model_i} - {opt_optimizer_i}')
        
        
        optimizer = initiate_optimizer(**opt_optimizer_new)
        
        # Load if previously saved
        tag = pickle.dumps(opt_model_new) + pickle.dumps(opt_optimizer_new) +             pickle.dumps(i_shape + o_shape) + pickle.dumps(PRECIP_XTRM)
        hashed_name = f'xtrm_{m_id}_{hashlib.md5(tag).hexdigest()}'
        tmp_file = pathlib.Path(f'tmp/{hashed_name}')
        
        if tmp_file.exists():
            custom_objects = {"weighted_cross_entropy_fn": weighted_binary_cross_entropy}
            with keras.utils.custom_object_scope(custom_objects):
                m = keras.models.load_model(tmp_file)
            
        else:
            # Create the model and compile
            m = DeepFactory(model, i_shape, o_shape, for_extremes=True, **opt_model_new)
            m.compile(
                loss=xtrm_loss,
                optimizer=optimizer
            )
            print(f'Number of parameters: {m.model.count_params()}')

            # Train
            hist = m.fit(dg_train, validation_data=dg_valid, verbose=history_log_level, **opt_training)
            
           
            # Saving the model
            m.save.weights(f'tmp/keras/{model}_xtrm.h5')
        
       
