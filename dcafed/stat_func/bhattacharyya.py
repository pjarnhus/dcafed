from scipy.stats import gaussian_kde
from scipy.integrate import romb
from pandas import DataFrame, concat, Series
from numpy import sqrt, log, issubdtype, number, log2, mgrid, complex


def _bhattacharyya_cat(df_1, df_2, var_list):
    """
    Utility function for calculating the Bhattacharyya distance between categorical columns in two DataFrames
    :param df_1: First DataFrame
    :param df_2: Second DataFrame
    :param var_list: list - categorical variables to be included in the calculation, all variables must match a column
    in each DataFrame
    :return: Pandas Series with the distance as values and column names as indices
    """

    # Calculate the ratio of every categorical variable in its column for both DataFrames
    p = df_1[var_list].melt().groupby(['variable', 'value']).size()/df_1.shape[0]
    q = df_2[var_list].melt().groupby(['variable', 'value']).size()/df_2.shape[0]

    # Reinsert missing values in the cases, where a value is only present in one of the two DataFrames
    has_missing = (p*q).isnull().groupby('variable').any()

    # Compute the Bhattacharyya coefficient and distance
    bc = (sqrt(p*q)).groupby('variable').sum()
    bc[has_missing] = None
    return -log(bc)


def _bhattacharyya_cont(df_1, df_2, var_list, continuous_integration_points):
    """
    Utility functino for calculating the Bhattacharyya distance between continuous columns in two DataFrames
    :param df_1: First DataFrame
    :param df_2: Second DataFrame
    :param var_list: list - continuous variables to be included in the calculation, all variables must match a column
    in each DataFrame
    :return: Pandas Series with the distance as values and column names as indices
    """

    assert (all([issubdtype(t, number) for t in df_1[var_list].dtypes])
            and all([issubdtype(t, number) for t in df_2[var_list].dtypes])),\
        'All continuous variables must be numerical'

    assert (log2(continuous_integration_points-1) % 1 == 0),\
        'The number of integration points must be 2**n+1 where n is a non-negative integer'
    # Compute the Gaussian KDEs for all continuous columns in both DataFrames
    kernels_1 = [gaussian_kde(df_1[v].get_values()) for v in var_list]
    kernels_2 = [gaussian_kde(df_2[v].get_values()) for v in var_list]

    # Define low and high points for each integration range, i.e. the end points of the two distributions
    low_points = [min(df_1[v].min(), df_2[v].min()) for v in var_list]
    high_points = [max(df_1[v].max(), df_2[v].max()) for v in var_list]

    # Calculate integration points
    positions = [mgrid[lp:hp:complex(0, continuous_integration_points)] for lp, hp in zip(low_points, high_points)]

    # Calculate the KDE values at each integration point
    kdes_1 = [k(p) for k, p in zip(kernels_1, positions)]
    kdes_2 = [k(p) for k, p in zip(kernels_2, positions)]

    # Return a pandas Series with the computed distances
    return Series([-log(romb(sqrt(k1*k2), dx=(p[1:]-p[:-1]).mean()))
                   for k1, k2, p in zip(kdes_1, kdes_2, positions)], index=var_list)


def bhattacharyya(df_1,
                  df_2,
                  categorical_variables=None,
                  continuous_variables=None,
                  continuous_integration_points=1025):
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
    :param continuous_integration_points: int - The number of points the integrals are evaluated at.
    Must be 2**n+1 where n is a non-negative integer
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
        if len(missing_var) > 0:
            cat_dist = concat([cat_dist, Series(None, index=missing_var)])

    if continuous_variables is not None:
        if isinstance(continuous_variables, str):
            var_list = [continuous_variables]
        else:
            var_list = continuous_variables

        missing_var = [v for v in var_list if (v not in df_1.columns) or (v not in df_2.columns)]
        var_list = [v for v in var_list if (v in df_1.columns) and (v in df_2.columns)]

        cont_dist = _bhattacharyya_cont(df_1, df_2, var_list, continuous_integration_points)
        if len(missing_var) > 0:
            cont_dist = concat([cont_dist, Series(None, index=missing_var)])

    return concat([cat_dist, cont_dist])
