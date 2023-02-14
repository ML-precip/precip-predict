# Important to note:!
# The use of innvestigate requires specific versions of python and keras
# The model must be built according to the versions, so the weights are saved separately
# import necesssary libaries
from pathlib import Path
import pathlib
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
#for plotting
import geopandas as gpd

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from utils.utils_unet import *
from utils.utils_plot import plot_map
from utils.DNN_models import *
from utils.utils_LRP import *

import innvestigate
import innvestigate.utils as iutils
import innvestigate.applications.imagenet
import innvestigate.utils.visualizations as ivis

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #catch FutureWarnings so that I can find actual errors!
    
import keras
import keras.backend as K
import keras.models
#import tensorflow as tf  # this is new for custom loss function
from keras.models import load_model
    
from keras.layers import Input, Dense, Activation, Dropout, Flatten, MaxPooling2D, Conv2D, Conv2DTranspose, UpSampling2D, Reshape
from keras.models import Model
    
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform



def save_rel(a, times, lats_y, lons_x,  PATH_OUT, method):
    """Args: a is the output from the LRP
       lat, lon, time from the training"""
    # Convert to xr.dataarray
    a_arr= xr.DataArray(a, dims=["time","lat", "lon", "var"],
                      coords=dict(time = times, lat = lats_y, 
                      lon = lons_x, var= conf['varnames'] ))
    
    f = PATH_OUT + method + '_train.nc'
    a_arr.to_netcdf(f)



import yaml
conf = yaml.safe_load(open("config.yaml"))

#save the test data
PATH_OUT = 'tmp/LRP/'
DIR_WEIGHTS = '/storage/homefs/no21h426/precip-predict/tmp/keras/'



#load the model configuration
# Define args for the U-net model



# load coordinates
lons_x = np.load('tmp/data/lons_y.npy')
lats_y = np.load('tmp/data/lats_y.npy')
# create a time array
times = np.arange(np.datetime64('1979-01-01'), np.datetime64('2006-01-01')) #until validation period
times = pd.to_datetime(times)
# load the training and testing data
dg_train_X = np.array(xr.open_dataarray('tmp/data/dg_train_X.nc'))
dg_train_Y = np.array(xr.open_dataarray('tmp/data/dg_train_Y.nc'))
#dg_train_Y_xtrm = np.array(xr.open_dataarray('tmp/data/dg_train_Y_xtrm.nc'))

#times = np.arange(np.datetime64('2016-01-01'), np.datetime64('2022-01-01')) #until validation period
#times = pd.to_datetime(times)
#dg_test_X = np.array(xr.open_dataarray('tmp/data/dg_test_X.nc'))

#dg_test_X_05 = np.ones_like(dg_test_X, dtype=float)*0.5
dg_train_X_05 = np.ones_like(dg_train_X, dtype=float)*0.5


i_shape = conf['i_shape']
o_shape = conf['o_shape']

print(f'X shape: {i_shape}')
print(f'y shape: {o_shape}')
output_channels = conf['output_channels']
num_filters = conf['num_filters']
use_batchnorm = conf['use_batchnorm']
dropout = conf['dropout']
lr = conf['lr']

name_model = conf['model']
output_scaling = 1
output_crop = None


# Define hyperparameters
EPOCHS = 10
LR_METHOD = 'Constant'  # Cyclical, CosineDecay, Constant
                                            
# Default model options
opt_model = {'latent_dim': 128,
             'dropout_rate': 0.2}

# Default optimizer options
opt_optimizer = {'lr_method': 'Constant',
                 'lr': 0.0004,
                 'init_lr': 0.01}

models = {
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


models_prec =[]
models_xtrm = []
# load weights and calculate LRP for the testing period
for m_id in models:
    
    if not models[m_id]['run']:
        continue
    print(m_id)
    # Extract model name and options
    model = models[m_id]['model']
    opt_model_i = models[m_id]['opt_model']
    opt_optimizer_i = models[m_id]['opt_optimizer']
    opt_model_new = opt_model.copy()
    opt_model_new.update(opt_model_i)
    opt_optimizer_new = opt_optimizer.copy()
    opt_optimizer_new.update(opt_optimizer_i)

    m = DeepFactory_Keras(model, i_shape, o_shape, for_extremes=False, for_lrp = True, **opt_model_new)
    m.model.load_weights(f'{DIR_WEIGHTS}ERA5-low_0.95_{m_id}.h5')
    
    print('loads weigths')
     
    models_prec.append(m)
    
    print('calculating LRP-comp for', m_id)
    
    print('analysing LRPcomp')
    # Use the innevestigate tool
    lrpcomp_test = calLRP(dg_train_X, m.model, 'comp', only_positive=False )
    print('saving LRP composite')
    np.save(f'tmp/LRP/lrpcomp_train_DNN_{m_id}.npy',lrpcomp_test)

    # Save baselines
    baseline_lrpcomp_test = calLRP(dg_train_X_05, m.model, 'comp', only_positive=False )
    print('saving LRP composite for baseline')
    np.save(f'tmp/LRP/baseline_lrpcomp_train_DNN_{m_id}.npy',baseline_lrpcomp_test)
    


####### Analyse different methods###########
#print('analysing LRPz')
# Use the innevestigate tool
#lrpz_test = calLRP(dg_test_X,m.model, 'lrpz', only_positive=False)
#print('saving LRPz')
#np.save('tmp/LRP/lrpz_test_DNN_UNET4.npy',lrpz_test)

#print('analysing LRPcomp')
# Use the innevestigate tool
#lrpcomp_test = calLRP(dg_test_X,m.model, 'comp', only_positive=False )
#print('saving LRP composite')
#np.save('tmp/LRP/lrpcomp_test_DNN_UNET4.npy',lrpcomp_test)

#print('analysing LRPcompflat')
# Use the innevestigate tool
#lrpcompflat_test = calLRP(dg_test_X,m.model, 'compflat', only_positive=False)
#print('saving LRPcompflat')
#np.save('tmp/LRP/lrpcompflat_test_DNN_UNET4.npy',lrpcompflat_test)


#gradient_LRP_test = calLRP(dg_test_X,m.model, 'gradient' )
#np.save('tmp/LRP/gradient_test_DNN_UNET4.npy',gradient_LRP_test)
#save relevances
#save_rel(aEp_test, times, lats_y, lons_x, PATH_OUT, 'Epsilon')
#print('analysing LRP alphabeta')
#a1b0_train= calLRP(dg_train_X,m.model, 'a1b0' )
#a1b0_test= calLRP(dg_test_X,m.model, 'a1b0' )
#np.save('tmp/LRP/a1b0_test_DNN_UNET4.npy',a1b0_test)
#print('saving LRP alphabeta')
#save_rel(a1b0_test, times, lats_y, lons_x, PATH_OUT, 'alphabeta')


