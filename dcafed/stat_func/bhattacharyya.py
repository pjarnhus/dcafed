from scipy.stats import gaussian_kde
from scipy.integrate import romb
from pandas import DataFrame, concat, Series
from numpy import sqrt, log


def _bhattacharyya_cat(df_1, df_2, var_list):
    """

    :param df_1: First DataFrame
    :param df_2: Second DataFrame
    :param var_list: list - categorical variables to be included in the calculation, all variables must match a column
    in each DataFrame
    :return: Pandas Series with the distance as values and column names as indices
    """
    p = df_1[var_list].melt().groupby(['variable', 'value']).size()/df_1.shape[0]
    q = df_2[var_list].melt().groupby(['variable', 'value']).size()/df_2.shape[0]

    has_missing = (p*q).isnull().groupby('variable').any()
    bc = (sqrt(p*q)).groupby('variable').sum()
    bc[has_missing] = None
    return -log(bc)


def _bhattacharyya_cont(df_1, df_2, var_list):
    return None

def bhattacharyya(df_1, df_2, categorical_variables=None, continuous_variables=None):
    """
    Compute the Bhattacharyya distance between distributions in the DataFrames df_1 and df_2
    Any column not in both DataFrames will be assumed to have infinity distance between the two distributions,
    and a missing value will be returned.
    A missing value is also returned if any categorical variable has a value in one DataFrame, that is not in the other
    The result is independent of the order of the DataFrames
    :param df_1: First DataFrame
    :param df_2: Second DataFrame
    :param categorical_variables: str or list - Either a single column name or a list of column names,
    indicating which columns should be treated as categorical
    :param continuous_variables: str or list - Either a single column name or a list of column names,
    indicating which columns should be treated as continuous
    :return: Pandas Series with the distance as values and column names as indices
    """

    assert isinstance(df_1, DataFrame), 'df_1 must be a pandas DataFrame'
    assert isinstance(df_2, DataFrame), 'df_2 must be a pandas DataFrame'
    assert isinstance(categorical_variables, str)\
        or isinstance(categorical_variables, list)\
        or categorical_variables is None,\
        'categorical_variables must be a string, a list or None'
    assert isinstance(continuous_variables, str)\
        or isinstance(continuous_variables, list)\
        or continuous_variables is None,\
        'continuous_variables must be a string, a list or None'

    if categorical_variables is None and continuous_variables is None:
        return None

    cat_dist = None
    cont_dist = None
    if categorical_variables is not None:
        if isinstance(categorical_variables, str):
            var_list = [categorical_variables]
        else:
            var_list = categorical_variables

        missing_var = [v for v in var_list if (v not in df_1.columns) or (v not in df_2.columns)]
        var_list = [v for v in var_list if (v in df_1.columns) and (v in df_2.columns)]
        cat_dist = _bhattacharyya_cat(df_1, df_2, var_list)
        if len(missing_var)>0:
            cat_dist = concat([cat_dist, Series(None, index=missing_var)])

    if continuous_variables is not None:
        if isinstance(continuous_variables, str):
            continuous_variables = continuous_variables

        cont_dist = _bhattacharyya_cont(df_1, df_2, continuous_variables)

    return concat([cat_dist, cont_dist])
