import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask
import datetime
import math





def getmap_rel(a, i_shape, y_bool, lons_x, lats_y, allmap =True):
    """Function to calculate the maximum relevances when extremes occur pr_xtrm==1
       Args: a: numpy array with the relevances
             i_shape: shape of the array
             y_bool: xarray.dataarry or array of the true extreme
             allmap: True, to calculate the maximum of relavances over the map, False for a pixel-wise calculation"""
    # process rel
    y_bool = np.array(y_bool)
    m = np.zeros(shape=(i_shape[0],i_shape[1], i_shape[2]))
    mmax = np.zeros(shape=(i_shape[0],i_shape[1], i_shape[2]))
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
                tmp[np.where(tmp < 0)] = 0.

            m[ilat, ilon,:] = tmp.mean(axis=0)
            mmax[ilat, ilon,:] = tmp.max(axis=0)
            
    return m, mmax


def getmap_localrel(a, i_shape, varnames, y_bool, lats_y, lons_x, times, icrop = 5):
    """Function to calculate the maximum relevances when extremes occur pr_xtrm==1
             over the closest pixels
             Args: a: numpy array with the relevances
             i_shape: shape of the array
             lats, lons: coordinates
             y_bool: xarray.dataarry or array of the true extreme
             icrop: select the number of pixels we want to calculate the relevances for a given point"""

    threshold_quant = np.quantile(a, 0.999)
    a_maxed = a
    a_maxed[a_maxed>threshold_quant] = threshold_quant
    
    m_a1b0= xr.DataArray(a, dims=["time","lat", "lon", "variable"],
                  coords=dict(time = times, lat = lats_y, 
            lon = lons_x, variable= varnames ))

    
    crop_lats = lats_y[icrop:len(lats_y)-icrop]
    crop_lons = lons_x[icrop:len(lons_x)-icrop]
        
    m = np.zeros(shape=(len(crop_lats),len(crop_lons), i_shape[2]))
    mmax = np.zeros(shape=(len(crop_lats),len(crop_lons), i_shape[2]))
    
    for ilon in range(0,len(crop_lons)):
        for ilat in range(0,len(crop_lats)):
            
            idx_s = y_bool.sel(lat=crop_lats[ilat], lon=crop_lons[ilon]) ==1
            idx_s = np.array(idx_s)
            idx_s = idx_s.flatten()
            
            m_sel = m_a1b0.sel(lat=slice(crop_lats[ilat]+icrop,crop_lats[ilat]-icrop),lon=slice(crop_lons[ilon]-icrop,crop_lons[ilon]+icrop))
            
            tmp = m_sel[idx_s, :, :, :]
            tmp = tmp.max(axis=(1,2))
           
            m[ilat, ilon,:] = tmp.mean(axis=0)
            mmax[ilat, ilon,:] = tmp.max(axis=0)
            
    return m, mmax