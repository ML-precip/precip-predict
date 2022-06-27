import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import geopandas as gpd
import itertools
import xarray as xr


def plot_prediction_scatter(y_test, y_pred):
    n_cols = 3
    n_plots = y_pred.shape[1]
    nrows = -(-n_plots // n_cols)

    fig, axs = plt.subplots(nrows=nrows, ncols=n_cols,
                            sharex=False, figsize=(17, 11))
    for i in range(n_plots):
        ax = plt.subplot(nrows, n_cols, i + 1)
        x = y_test.iloc[:, i]
        y = y_pred[:, i]
        ax.scatter(x, y, facecolors='none', edgecolors='k', alpha=0.5)
        v_max = max(x.max(), y.max())
        ax.plot([0, v_max], [0, v_max], 'r--')
        ax.set_xlabel('Observed prec. [mm]')
        ax.set_ylabel('Predicted prec. [mm]')
        ax.set_title(y_test.iloc[:, i].name)
        ax.set_xlim([x.min(), 1.05 * v_max])
        ax.set_ylim([y.min(), 1.05 * v_max])


def plot_prediction_ts(test_dates, final_predictions, test_labels):
    df_to_compare = pd.DataFrame(
        {'date': test_dates, 'Actual': test_labels, 'Predicted': final_predictions})
    dfm = pd.melt(df_to_compare, id_vars=['date'], value_vars=[
                  'Actual', 'Predicted'], var_name='data', value_name='precip')
    f, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sns.regplot(data=df_to_compare, x="Actual", y="Predicted", ax=axs[0], )
    sns.lineplot(x='date', y='precip', hue='data', data=dfm, ax=axs[1])


def plot_importance(features_importance, attributes, IMAGES_PATH):
    indices = np.argsort(features_importance)
    plt.barh(range(len(attributes)),
             features_importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [attributes[i] for i in indices])
    plt.xlabel('Relative Importance')
    save_fig("Rela_Importance", IMAGES_PATH)
    plt.show()


def save_fig(fig_id, IMAGES_PATH, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_hist(history):
    
    # plot the train and validation losses
    N = np.arange(len(history.history['loss']))
    plt.figure()
    plt.plot(N, history.history['loss'], label='train_loss')
    plt.plot(N, history.history['val_loss'], label='val_loss')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='upper right')
    
    plt.show()

    
def plot_map(ax, lons, lats, vals, title=None, vmin=None, vmax=None, cmap=None, show_colorbar=True):
    """ Plotting a map with the provided values and the country boundaries."""
    
    # Load country outlines
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    im = ax.pcolormesh(lons, lats, vals, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    world.boundary.plot(ax=ax, lw=1, color='k')
    ax.set_xlim(min(lons), max(lons))
    ax.set_ylim(min(lats), max(lats))
    if title:
        ax.set_title(title)
    if show_colorbar:
        # adding more settings to the colorbar to control aspect
        plt.colorbar(im, ax=ax, shrink=.5, pad=.1, aspect=8)

        
        
def plot_relevances(rel):
    """ plot relevances for each variable, e.g., 31 (5 variables * 6 levels + 1 tpcw)
        Args: a matrix with the relevances calculated based on a specific method (e.g. LRP)
        """
    n_figs = rel.shape[2]
    ncols = 5
    nrows = -(-n_figs // ncols)
    fig, axes = plt.subplots(figsize=(24, 3.2*nrows), ncols=ncols, nrows=nrows)
    # set title 
    n_tit = list(itertools.product(conf['variables'][:-1], conf['levels']))
    n_tit.append(conf['variables'][5])
    for i in range(n_figs):
        i_row = i // ncols
        i_col = i % ncols
        ax = axes[i_row, i_col]
        vals = rel[:,:,i]
        plot_map(ax, lons_x, lats_y, vals, title=str(n_tit[i]))


    
def plot_xr_rel(rel, lats_y,lons_x, vnames, fname, cmap='Reds', plot=True):
    
    
    mx= xr.DataArray(rel, dims=["lat", "lon", "variable"],
                  coords=dict(lat = lats_y, 
            lon = lons_x, variable= vnames ))
    
    g = mx.plot.pcolormesh("lon", "lat", col="variable", col_wrap=4, robust=True, cmap=cmap,
    yincrease = False, extend='max',
    figsize=(14, 14),  cbar_kwargs={"orientation": "vertical", "shrink": 0.9, "aspect": 50})
    #figsize=(14, 12)
    for ax, title in zip(g.axes.flat, vnames):

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world.boundary.plot(ax=ax, lw=1, color='k')
        ax.set_xlim(min(lons_x), max(lons_x))
        ax.set_title(title)
        ax.set_ylim(min(lats_y), max(lats_y))
        
    # To control the space
    plt.subplots_adjust(right=0.8, wspace=-0.6, hspace=0.2)
    if plot:
        plt.savefig('figures/' + fname + '.pdf')
    else:
        
        plt.draw()