#!/usr/bin/env python
# coding: utf-8

# Important to note:!
# The use of innvestigate requires specific versions of python and keras
# The model must be built according to the versions, so the weights are saved separately
# import necesssary libaries

# Calculate LRP for U-Net 

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


import yaml
conf = yaml.safe_load(open("config.yaml"))


#load the model again
# Define args for the U-net model
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


u_mod = Unet2(name_model, i_shape, o_shape, output_channels, num_filters, False, dropout)



if (name_model == 'unet'):
    model = u_mod.build_model()
elif (name_model == 'unet-att'):
    model = u_mod.att_unet()
elif (name_model == 'unet-convlstm'):
    model = u_mod.build_UnetConvLSTM()




model.compile(loss=keras.losses.categorical_crossentropy, ## instead of CategoricalCrossentropy
                  optimizer='adam', ## lr instead of learning_rate
                  metrics=['categorical_accuracy'])




f_weights = 'tmp/tmp_weights/' + name_model + '_trained_weights.h5'



model.load_weights(f_weights)


# load data
# coordinates
lons_x = np.load('tmp/data/lons_y.npy')
lats_y = np.load('tmp/data/lats_y.npy')

# test samples
dg_test_X = np.array(xr.open_dataarray('tmp/data/dg_test_X.nc'))
dg_test_Y = np.array(xr.open_dataarray('tmp/data/dg_test_Y.nc'))
dg_test_Y_xtrm = np.array(xr.open_dataarray('tmp/data/dg_test_Y_xtrm.nc'))

dg_train_X = np.array(xr.open_dataarray('tmp/data/dg_train_X.nc'))
dg_train_Y = np.array(xr.open_dataarray('tmp/data/dg_train_Y.nc'))
dg_train_Y_xtrm = np.array(xr.open_dataarray('tmp/data/dg_train_Y_xtrm.nc'))

#load predictions --numpy array
y_pred_test = np.load('tmp/data/y_pred_test.npy') 
y_pred_bool = y_pred_test >= 0.5
y_pred_bool = y_pred_bool * 1

# test-times
times = np.arange(np.datetime64('2016-01-01'), np.datetime64('2021-12-31'))
times = pd.to_datetime(times)



nx = dg_test_X.shape[1]
ny = dg_test_X.shape[2]
nchans = dg_test_X.shape[3]


#LRP algorithm using Alpha-Beta rule - Alpha1Beta0 only tracks positive relevance
lrp_analyzerA1B0 = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0(model)



a1b0 = lrp_analyzerA1B0.analyze(dg_train_X)

np.save('tmp/LRP/a_train_a1b0_myUNET.npy',a1b0)


