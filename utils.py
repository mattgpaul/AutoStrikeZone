import sys
import pandas as pd

sys.path.append('..')


def Normalize(df, axis=1):
    """Normalize a dataframe along an axis.

    Args:
        df (pd.DataFrame): Pandas dataframe of size mxn.
        axis (int, optional): axis to perform statistics operations. Defaults to 1.

    Returns:
        df_norm (pd.DataFrame): Returns a dataframe with the same size as the input. Dataframe values are now formated as (df - mean)/std for each entry along the user specified axis
        df_mu (float): mean of the values along the user specified axis
        df_sigma (float): standard deviation of the values along the user specified axis
    """

    df_mu = df.mean(axis=axis)
    df_sigma = df.std(axis=axis)
    df_norm = (df - df_mu)/df_sigma

    return df_norm, df_mu, df_sigma