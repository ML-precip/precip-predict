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
        #analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0(model)
        analyzer = innvestigate.create_analyzer('lrp.alpha_beta', model, alpha=1,beta=0)
    elif lrpRule == 'epsilon':
        analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(model)
    elif lrpRule == 'lrpz':
        #analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZ(model)
        analyzer = innvestigate.create_analyzer('lrp.z', model)
    elif lrpRule == 'gradient':
        analyzer = innvestigate.create_analyzer('gradient', model)
    elif lrpRule == 'deep_taylor':
        analyzer = innvestigate.create_analyzer('deep_taylor', model)
    elif lrpRule == 'comp':
        analyzer = innvestigate.create_analyzer('lrp.sequential_preset_a', model)
    elif lrpRule == 'compflat':
        analyzer = innvestigate.create_analyzer('lrp.sequential_preset_a_flat', model)

        
    
    ### Analyze each input via the analyzer
    for i in np.arange(0,np.shape(X)[0]):
            sample = X[i]
            analyzer_output = analyzer.analyze(sample[np.newaxis,...])
            # no need to normalise
            deepMaps[i] = analyzer_output
            #deepMaps[i] = analyzer_output/np.sum(analyzer_output.flatten())

    ### Save only the positive contributions
            if only_positive:
                deepMaps[np.where(deepMaps < 0)] = 0.
           
    return deepMaps


