# precip-predict
Precipitation prediction with Machine Learning

This repository contains all the code for processing the data as well as code for the DL and RF models used in the paper Otero and Horton, 2022: 
https://eartharxiv.org/repository/view/3447/


A total of 6 deep learning architectures are tested to predict both, precipitation amounts and extreme precipitation events (>95th and >99th). A baseline model is also include to benchmark the DL performance.

* DNN_models_comparison.ipynb: Contains the DeepFactory class with all the models architectures.
* Random_Forest.ipynb: Runs RF regressor and classifier point-wise.

In addition, a layer-wise-relevance propagation (LRP) is applied to assess the importance of the predictors.

The data used for this study can be download from https://cds.climate.copernicus.eu/cdsapp\#!/dataset/reanalysis-era5-pressure-levels.

