import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

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