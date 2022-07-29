def Normalize(df, axis=1):

    df_mu = df.mean(axis=axis)
    df_sigma = df.std(axis=axis)
    df_norm = (df - df_mu)/df_sigma

    return df_norm, df_mu, df_sigma