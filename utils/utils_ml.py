from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as tfb
from collections import OrderedDict
import tensorflow.keras.backend as K
from sklearn.utils import class_weight
from sklearn.utils import shuffle



# From https://developpaper.com/tensorflow-chapter-tensorflow-2-x-local-training-and-evaluation-based-on-keras-model/
class WeightedBinaryCrossEntropy(keras.losses.Loss):
    """
    Args:
    pos_weight: Scalar to affect the positive labels of the loss function.
    weight: Scalar to affect the entirety of the loss function.
    from_logits: Whether to compute loss from logits or the probability.
    reduction: Type of tf.keras.losses.Reduction to apply to loss.
    name: Name of the loss function.
    """

    def __init__(self,
                 pos_weight,
                 weight,
                 from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.losses.binary_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
        )[:, None]
        ce = self.weight * (ce * (1 - y_true) + self.pos_weight * ce * y_true)

        return ce


def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    From https://helioml.org/08/notebook.html
    """
    # Multiplier for positive targets, needs to be tuned
    POS_WEIGHT = 5

    # Transform back to logits
    _epsilon = tf.convert_to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tfb.log(output / (1 - output))

    # Compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)

    return tf.reduce_mean(loss, axis=-1)
    
    
def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False):
    """
    credits: https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy
    Return a function for calculating weighted binary cross entropy
    It should be used for multi-hot encoded labels
    @param weights A dict setting weights for 0 and 1 label. e.g.
        {
            0: 1.
            1: 8.
        }
        For this case, we want to emphasise those true (1) label, 
        because we have many false (0) label. e.g. 
            [
                [0 1 0 0 0 0 0 0 0 1]
                [0 0 0 0 1 0 0 0 0 0]
                [0 0 0 0 1 0 0 0 0 0]
            ]



    @param from_logits If False, we apply sigmoid to each logit
    @return A function to calcualte (weighted) binary cross entropy
    '''
    """
    assert 0 in weights
    assert 1 in weights

    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = K.mean(tf.multiply(ce, weights_v))
        return loss

    return weighted_cross_entropy_fn


def split_data(df, yy_train, yy_test, attributes, ylabel):
    """"Split the data into train and test
         df is the data\n",
         attributes are the covariates,
         ylabel is the target variable"""
    train_dataset = df[(df.date.dt.year >= yy_train[0]) &
                       (df.date.dt.year <= yy_train[1])]
    test_dataset = df[(df.date.dt.year >= yy_test[0]) &
                      (df.date.dt.year <= yy_test[1])]

    # Extract the dates for each datasets
    train_dates = train_dataset['date']
    test_dates = test_dataset['date']

    # Extract labels
    train_labels = train_dataset[ylabel].copy()
    test_labels = test_dataset[ylabel].copy()
    
    # Extract predictors\n",
    train_dataset = train_dataset[attributes]
    test_dataset = test_dataset[attributes]

    return(train_dataset, train_labels, test_dataset, test_labels, train_dates, test_dates)


def create_pipeline(data, cat_var):
    """Prepare the data in the right format for the model"""

    num_attribs = list(data)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    if (cat_var != None):
        num_attribs.remove(cat_var)
        cat_attribs = [cat_var]
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
    else:
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
        ])

    return(full_pipeline)


def evaluate_model(test_labels, train_labels, predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')


class DataGenerator(keras.utils.Sequence):
    # credits: https://github.com/pangeo-data/WeatherBench/blob/master/notebooks/3-cnn-example.ipynb
    def __init__(self, ds, var_dict, batch_size=32, shuffle=True, load=True, mean=None, std=None):
    #def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, mean=None, std=None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours # I am skipping it for now
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """
        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        #self.lead_time = lead_time

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            #if var=="T2MMEAN":
            if levels is None:
                data.append(ds[var].expand_dims({'level': generic_level}, 1)) 
            else:
                data.append(ds[var])
    
   
        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
     
        # Normalize
        self.data = (self.data - self.mean) / self.std
        #self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        #self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        #self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        #self.on_epoch_end()
        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    
class MyDataGenerator(keras.utils.Sequence):
    # adapted from: https://github.com/pangeo-data/WeatherBench/blob/master/notebooks/3-cnn-example.ipynb
    def __init__(self, ds, labels, var_dict, batch_size=32, shuffle=True, load=True, mean=None, std=None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Adapted by https://github.com/pangeo-data/WeatherBench/blob/master/src/train_nn.py
        Args:
            ds: Dataset containing all variables
            # change
            labels: predictand i.e. Target variable. It must be np.array
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """
        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.labels = labels


        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            if levels is None:
                data.append(ds[var].expand_dims({'level': generic_level}, 1)) 
            else:
                data.append(ds[var])

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std

        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.shape[0]
        #self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        #self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        #self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: 
            print('Loading data into RAM') 
            self.data.load()
            self.labels.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        y = self.labels.isel(time=idxs).values
        #y = self.labels[idxs,:,:]
        #y = self.data.isel(time=idxs + self.lead_time).values
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


class WeatherDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, var_dict, batch_size=32, shuffle=True, load=True, mean=None, std=None):
        """
        Data generator class.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Adapted by https://github.com/pangeo-data/WeatherBench/blob/master/src/train_nn.py
        Args:
            X: Dataset containing all predictor variables
            y: Dataset containing the variable
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            if levels is None:
                data.append(X[var].expand_dims({'level': generic_level}, 1)) 
            else:
                data.append(X[var])

        self.X = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.X.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.X.std('time').mean(('lat', 'lon')).compute() if std is None else std

        # Normalize
        self.X = (self.X - self.mean) / self.std
        self.n_samples = self.X.shape[0]

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: 
            print('Loading data into RAM')
            self.X.load()
            self.y.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.X.isel(time=idxs).values
        y = self.y.isel(time=idxs).values
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


            
# From https://github.com/pangeo-data/WeatherBench/blob/master/src/score.py
            
            
def datanormalise(ds, yy, var_dict, mean=None, std=None, shuf=False, extend_dim = False):
    """Function to normalise the inputs
       Args:
       ds is the list with the predictors
       yy is the labels
       var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
       mean: If None, compute mean from data.
       std: If None, compute standard deviation from data.
       shuffle: if True, data is shuffled
       extend_dim: if one more dimension is wanted
        """
    data = []
    generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
    for var, levels in var_dict.items():
        #if var=="T2MMEAN":
        if levels is None:
            data.append(ds[var].expand_dims({'level': generic_level}, 1)) 
        else:
            data.append(ds[var])
    
    data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
    mean = data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
    std = data.std('time').mean(('lat', 'lon')).compute() if std is None else std
    # Normalize
    data = (data - mean) / std
    y = np.array(yy)
    np_data = np.array(data)

    
    if shuf == True:
        np_data = shuffle(np_data)
        y = shuffle(y)
        
    if extend_dim ==True:
        # This is for the CONVLSTM
        # Expand the dimensions for convLSTM as it expects a 5D tensor
        np_data = tf.expand_dims(np_data, -1)
        y = tf.expand_dims(y, -1)    
    
    return np_data, y, mean, std
    
    
            
def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    
# From https://github.com/pangeo-data/WeatherBench/blob/master/src/score.py
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse


def get_rmse(truth, pred):
    """Function to calculate the RMSE
       It is similar to compute_weighted_rmse"""
    weights = np.cos(np.deg2rad(truth.lat))
    rmse = np.sqrt(((truth - pred)**2).weighted(weights).mean(['lat', 'lon']))
    return rmse


def compute_weighted_acc(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    
    
   # From https://github.com/pangeo-data/WeatherBench/blob/master/src/score.py

    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    clim = da_true.mean('time')
    try:
        t = np.intersect1d(da_fc.time, da_true.time)
        fa = da_fc.sel(time=t) - clim
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            np.sum(w * fa_prime * a_prime) /
            np.sqrt(
                np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
            )
    )
    return acc

def compute_weighted_mae(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the MAE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        mae: Latitude weighted root mean absolute error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    mae = (np.abs(error) * weights_lat).mean(mean_dims)
    return mae
