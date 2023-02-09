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