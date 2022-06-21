from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
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


def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed



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


class DataGeneratorWithExtremes(keras.utils.Sequence):
    def __init__(self, X, y, y_xtrm, var_dict, batch_size=32, shuffle=True, load=True, mean=None, std=None):
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
        self.y_xtrm = y_xtrm
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.for_xtrm = False

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            if levels is None:
                data.append(X[var].expand_dims({'level': generic_level}, 1)) 
            else:
                data.append(X[var].sel(level=levels))

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
            self.y_xtrm.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.X.isel(time=idxs).values
        if self.for_xtrm:
            y = self.y_xtrm.isel(time=idxs).values
        else:
            y = self.y.isel(time=idxs).values
        return X, y

    def for_extremes(self, val=True):
        self.for_xtrm = val

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


def eval_confusion_matrix_on_map(y_true, y_pred):
    """Compute the confusion matrix values for each point of the map"""
    tn = np.zeros(y_pred.shape[1:3])
    fp = np.zeros(y_pred.shape[1:3])
    fn = np.zeros(y_pred.shape[1:3])
    tp = np.zeros(y_pred.shape[1:3])
    for i_lat in range(y_pred.shape[1]):
        for i_lon in range(y_pred.shape[2]):
            tn[i_lat, i_lon], fp[i_lat, i_lon], fn[i_lat, i_lon], tp[i_lat, i_lon] = confusion_matrix(y_true[:, i_lat, i_lon], y_pred[:, i_lat, i_lon]).ravel()

    return tn, fp, fn, tp


def eval_confusion_matrix_scores_on_map(y_true, y_pred, manual=False):
    """Compute the precision and recall values for each point of the map"""
    precision_matrix = np.zeros(y_pred.shape[1:3])
    recall_matrix = np.zeros(y_pred.shape[1:3])
    for i_lat in range(y_pred.shape[1]):
        for i_lon in range(y_pred.shape[2]):
            if manual:
                tn, fp, fn, tp = confusion_matrix(y_true[:, i_lat, i_lon], y_pred[:, i_lat, i_lon]).ravel()
                precision_matrix[i_lat, i_lon] = tp / (tp + fp)
                recall_matrix[i_lat, i_lon] = tp / (tp + fn)
            else:
                precision_matrix[i_lat, i_lon] = precision_score(y_true[:, i_lat, i_lon], y_pred[:, i_lat, i_lon], zero_division=0)
                recall_matrix[i_lat, i_lon] = recall_score(y_true[:, i_lat, i_lon], y_pred[:, i_lat, i_lon])
    
    return precision_matrix, recall_matrix


def eval_roc_auc_score_on_map(y_true, y_probs, manual=False):
    """Compute the ROC AUC values for each point of the map"""
    roc_auc_matrix = np.zeros(y_probs.shape[1:3])
    for i_lat in range(y_probs.shape[1]):
        for i_lon in range(y_probs.shape[2]):
            roc_auc_matrix[i_lat, i_lon] = roc_auc_score(y_true[:, i_lat, i_lon], y_probs[:, i_lat, i_lon])
    
    return roc_auc_matrix





def get_scores(true, pred, scores):

        """function to get different socres
         Args: true , observations
               pred, predictions
        It is adapted from pySTEPS/pysteps/blob/ba2c81195dd36d1bede5370e0c5c1f4420657d6e/pysteps/verification/detcatscores.py
               
        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  ACC       | accuracy (proportion correct)                          |
        +------------+--------------------------------------------------------+
        |  BIAS      | frequency bias                                         |
        +------------+--------------------------------------------------------+
        |  CSI       | critical success index (threat score)                  |
        +------------+--------------------------------------------------------+
        |  ETS       | equitable threat score                                 |
        +------------+--------------------------------------------------------+
        |  F1        | the harmonic mean of precision and sensitivity         |
        +------------+--------------------------------------------------------+
        |  FA        | false alarm rate (prob. of false detection, fall-out,  |
        |            | false positive rate)                                   |
        +------------+--------------------------------------------------------+
        |  FAR       | false alarm ratio (false discovery rate)               |
        +------------+--------------------------------------------------------+
        |  GSS       | Gilbert skill score (equitable threat score)           |
        +------------+--------------------------------------------------------+
        |  HK        | Hanssen-Kuipers discriminant (Pierce skill score)      |
        +------------+--------------------------------------------------------+
        |  HSS       | Heidke skill score                                     |
        +------------+--------------------------------------------------------+
        |  MCC       | Matthews correlation coefficient                       |
        +------------+--------------------------------------------------------+
        |  POD       | probability of detection (hit rate, sensitivity,       |
        |            | recall, true positive rate)                            |
        +------------+--------------------------------------------------------+
        |  SEDI      | symmetric extremal dependency index                    |
        +------------+--------------------------------------------------------+
        """


        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

        H = tp  # true positives
        M = fn # false negatives
        F = fp  # false positives
        R = tn # true negatives

        result = {}
        for score in scores:
            # catch None passed as score
            if score is None:
                continue
            score_ = score.lower()

            # simple scores
            POD = H / (H + M)
            FAR = F / (H + F)
            FA = F / (F + R)
            s = (H + M) / (H + M + F + R)

            if score_ in ["pod", ""]:
                # probability of detection
                result["POD"] = POD
            if score_ in ["far", ""]:
                # false alarm ratio
                result["FAR"] = FAR
            if score_ in ["fa", ""]:
                # false alarm rate (prob of false detection)
                result["FA"] = FA
            if score_ in ["acc", ""]:
                # accuracy (fraction correct)
                ACC = (H + R) / (H + M + F + R)
                result["ACC"] = ACC
            if score_ in ["csi", ""]:
                # critical success index
                CSI = H / (H + M + F)
                result["CSI"] = CSI
            if score_ in ["bias", ""]:
                # frequency bias
                B = (H + F) / (H + M)
                result["BIAS"] = B

            # skill scores
            if score_ in ["hss", ""]:
                # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
                HSS = 2 * (H * R - F * M) / ((H + M) * (M + R) + (H + F) * (F + R))
                result["HSS"] = HSS
            if score_ in ["hk", ""]:
                # Hanssen-Kuipers Discriminant
                HK = POD - FA
                result["HK"] = HK
            if score_ in ["gss", "ets", ""]:
                # Gilbert Skill Score
                GSS = (POD - FA) / ((1 - s * POD) / (1 - s) + FA * (1 - s) / s)
                if score_ == "ets":
                    result["ETS"] = GSS
                else:
                        result["GSS"] = GSS
            if score_ in ["sedi", ""]:
                # Symmetric extremal dependence index
                SEDI = (np.log(FA) - np.log(POD) + np.log(1 - POD) - np.log(1 - FA)) / (
                    np.log(FA) + np.log(POD) + np.log(1 - POD) + np.log(1 - FA)
                )
                result["SEDI"] = SEDI
            if score_ in ["mcc", ""]:
                # Matthews correlation coefficient
                MCC = (H * R - F * M) / np.sqrt((H + F) * (H + M) * (R + F) * (R + M))
                result["MCC"] = MCC
            if score_ in ["f1", ""]:
                # F1 score
                F1 = 2 * H / (2 * H + F + M)
                result["F1"] = F1

        return result
    
    
    
    
def score_matrix(y_true, y_probs, name_score):
    
    """function to get a matrix-score
        Args: y_true: observations
              y_probs: prediction
              name_score: the score to be estimated"""
    
    smatrix = np.zeros(y_probs.shape[1:3])
    for i_lat in range(y_probs.shape[1]):
        for i_lon in range(y_probs.shape[2]):
            temp = get_scores(y_true[:, i_lat, i_lon], y_probs[:, i_lat, i_lon], [name_score])
            smatrix[i_lat, i_lon] = temp[name_score]
    return smatrix