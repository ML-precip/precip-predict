import innvestigate
import innvestigate.utils as iutils
import innvestigate.applications.imagenet
import innvestigate.utils.visualizations as ivis

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #catch FutureWarnings so that I can find actual errors!
    
import keras
import keras.backend as K
import xarray as xr
import numpy as np
import pandas as pd


def calLRP(X, model, lrpRule, only_positive=True):
    """Calculate the LRP based on the innvestigate available methods
       Args: X array-4D
             lrpRule: method to apply"""
    
    deepMaps = np.empty(np.shape(X))
    deepMaps[:] = np.nan
    # Inizialize the method
    
    if lrpRule == 'a1b0':
        analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0(model)
    elif lrpRule == 'epsilon':
        analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(model)
    elif lrpRule == 'gradient':
        analyzer = innvestigate.create_analyzer('gradient', model)
    elif lrpRule == 'deep_taylor':
        analyzer = innvestigate.create_analyzer('deep_taylor', model)
        
    
    ### Analyze each input via the analyzer
    for i in np.arange(0,np.shape(X)[0]):
            sample = X[i]
            analyzer_output = analyzer.analyze(sample[np.newaxis,...])
            deepMaps[i] = analyzer_output/np.sum(analyzer_output.flatten())

    ### Save only the positive contributions
            if only_positive:
                deepMaps[np.where(deepMaps < 0)] = 0.
           
    return deepMaps




def getmap_rel(a, i_shape, y_bool, allmap =True):
    # process rel
    m = np.zeros(shape=(i_shape[0],i_shape[1], i_shape[2]))
    for ilon in range(0,len(lons_x)):
        for ilat in range(0,len(lats_y)):
            idx_s = y_bool[:,ilat, ilon] ==1
            idx_s = idx_s.flatten()
            # select the extreme dates and calculate the averages
            if (allmap == True):
                tmp = a[idx_s, :, :, :]
                tmp = tmp.max(axis=(1,2))
            else:
                tmp = a[idx_s, ilat, ilon, :]

            m[ilat, ilon,:] = tmp.mean(axis=0)
            
    return(m)