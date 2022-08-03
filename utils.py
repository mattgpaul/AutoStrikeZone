from optparse import Values
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append('..')


def Normalize(df, axis=0):
    """Normalize a dataframe along an axis.

    Args:
        df (pd.DataFrame): Pandas dataframe of size mxn.
        axis (int, optional): axis to perform statistics operations. Defaults to 0.

    Returns:
        df_norm (pd.DataFrame): Returns a dataframe with the same size as the input. Dataframe values are now formated as (df - mean)/std for each entry along the user specified axis
        df_mu (float): mean of the values along the user specified axis
        df_sigma (float): standard deviation of the values along the user specified axis
    """

    df_mu = df.mean(axis=axis)
    df_sigma = df.std(axis=axis)
    df_norm = (df - df_mu)/df_sigma

    return df_norm, df_mu, df_sigma


def CreateFeatureColumns(df, features):
    """Creates tensorflow feature columns

    Args:
        features (list): columns in a dataframe that need to be transferred to feature columns

    Returns:
        feature_columns (tensorflow): numeric feature columns from the dataframe
    """

    feature_columns = []
    for feature in features:
        feature_columns.append(tf.feature_column.numeric_column(feature))

    feature_layer = layers.DenseFeatures(feature_columns)
    feature_layer(dict(df))

    return feature_layer



def PlotDecisionBoundary(clf, X, y):
    plots = []
    for label in y.unique():
        xplot = X[X.columns[0]][y == label]
        yplot = X[X.columns[1]][y == label]

        trace = go.Scatter(
            x=xplot,
            y=yplot,
            mode='markers',
            name=str(label)
        )
        plots.append(trace)

    # Create the meshgrid
    feature_x = np.arange(X[X.columns[0]].min(), X[X.columns[0]].max(), 0.1)
    feature_y = np.arange(X[X.columns[1]].min(), X[X.columns[1]].max(), 0.1)

    xx, yy = np.meshgrid(feature_x, feature_y)
    mesh = pd.DataFrame(np.reshape(np.stack((xx.ravel(), yy.ravel()), axis=1),(-1,2)), columns = [X.columns[0], X.columns[1]])

    zz = clf.predict(mesh)
    
    plots.append(go.Contour(
        x=feature_x, 
        y=feature_y, 
        z=np.reshape(zz,(xx.shape[0],-1)),
        contours_coloring='lines',
        line_width=3,
        hoverinfo='skip',
        contours=dict(
            start=y.min(),
            end=y.max(),
            size=len(y.unique())
        )
    ))

    layout = go.Layout(template='plotly_dark',title='Decision Boundary')
    fig = go.Figure(data=plots, layout=layout)
    return fig