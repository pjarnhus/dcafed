from scipy.stats import gaussian_kde
from scipy.integrate import romb
from pandas import DataFrame


def bhattacharyya(df_1, df_2, categorical_variables=None, continuous_variables=None):
    """
    Compute the Bhattacharyya distance between distributions in the DataFrames df_1 and df_2
    Any column not in both DataFrames will be assumed to have infinity distance between the two distributions,
    and a missing value will be returned.
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
