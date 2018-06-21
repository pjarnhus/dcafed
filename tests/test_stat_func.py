import unittest
import pandas as pd
import numpy as np
from dcafed import stat_func as sf


class StatFuncTests(unittest.TestCase):
    # executed prior to each test
    def setUp(self):
        np.random.seed(42)
        s = 100
        self.df_1 = pd.DataFrame({'cat_same_dist': [0, 1] * int(s/2),
                                  'cat_diff_dist': [0] * int(s/3) + [1] * (s - int(s / 3)),
                                  'cat_diff_vals': [0] * int(s / 3) + [1] * int(s / 3) + [2] * (s - 2*int(s/3)),
                                  'cont_same_dist': np.random.randn(s),
                                  'cont_diff_dist': np.random.randn(s) + 1,
                                  'cont_not_in_both': np.random.randn(s)})

        self.df_2 = pd.DataFrame({'cat_same_dist': [0, 1] * int(s / 2),
                                  'cat_diff_dist': [1] * int(s / 3) + [0] * (s - int(s / 3)),
                                  'cat_diff_vals': [0, 1] * int(s / 2),
                                  'cat_not_in_both': np.random.randint(50, size=s),
                                  'cont_same_dist': np.random.randn(s),
                                  'cont_diff_dist': np.random.randn(s) - 1})

        self.str_1 = 'abc'
        self.str_2 = 'def'

    # executed after every test
    def tearDown(self):
        pass

    # General input tests
    def test_bhattacharyya_missing_no_input(self):
        with self.assertRaises(TypeError):
            sf.bhattacharyya()

    def test_bhattacharyya_missing_one_input(self):
        with self.assertRaises(TypeError):
            sf.bhattacharyya()

    def test_bhattacharyya_wrong_type_df_1(self):
        with self.assertRaises(AssertionError) as ctx:
            sf.bhattacharyya(self.str_1, self.df_2)
        self.assertEqual("df_1 must be a pandas DataFrame", str(ctx.exception))

    def test_bhattacharyya_wrong_type_df_2(self):
        with self.assertRaises(AssertionError) as ctx:
            sf.bhattacharyya(self.df_1, self.str_2)
        self.assertEqual("df_2 must be a pandas DataFrame", str(ctx.exception))

    def test_bhattacharyya_wrong_type_categorical_variables(self):
        with self.assertRaises(AssertionError) as ctx:
            sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=123)
        self.assertEqual("categorical_variables must be a string, a list or None", str(ctx.exception))

    def test_bhattacharyya_wrong_type_continuous_variables(self):
        with self.assertRaises(AssertionError) as ctx:
            sf.bhattacharyya(self.df_1, self.df_2, continuous_variables=123)
        self.assertEqual("continuous_variables must be a string, a list or None", str(ctx.exception))

    def test_bhattacharyya_no_columns_passed(self):
        self.assertIsNone(sf.bhattacharyya(self.df_1, self.df_2))

    def test_bhattacharyya_wrong_number_of_integration_points(self):
        cont = ['cont_same_dist', 'cont_diff_dist']
        with self.assertRaises(AssertionError) as ctx:
            d = sf.bhattacharyya(self.df_1, self.df_2, continuous_variables=cont, continuous_integration_points=1024)
        self.assertEqual('The number of integration points must be 2**n+1 where n is a non-negative integer',
                         str(ctx.exception))

    # Categorical value tests
    def test_bhattacharyya_cat_distance(self):
        var = ['cat_same_dist', 'cat_diff_dist']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=var)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(all([v in d.index for v in var]+[v in var for v in d.index]))
        self.assertEqual(0.0, d['cat_same_dist'])
        self.assertEqual(0.0614, np.round(d['cat_diff_dist'], decimals=4))

    def test_bhattacharyya_cat_different_values(self):
        var = ['cat_same_dist', 'cat_diff_dist', 'cat_diff_vals']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=var)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(all([v in d.index for v in var] + [v in var for v in d.index]))
        self.assertTrue(d.isnull()['cat_diff_vals'])

    def test_bhattacharyya_cat_str_input(self):
        var = 'cat_same_dist'
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=var)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(var in d.index)
        self.assertEqual(0.0, d[var])

    def test_bhattacharyya_cat_column_not_in_both(self):
        var = ['cat_same_dist', 'cat_diff_dist', 'cat_not_in_both']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=var)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(all([v in d.index for v in var] + [v in var for v in d.index]))
        self.assertTrue(d.isnull()['cat_not_in_both'])

    # Continuous value tests
    def test_bhattacharyya_cont_distance(self):
        var = ['cont_same_dist', 'cont_diff_dist']
        d = sf.bhattacharyya(self.df_1, self.df_2, continuous_variables=var)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(all([v in d.index for v in var]+[v in var for v in d.index]))
        self.assertEqual(0.0138, np.round(d['cont_same_dist'], decimals=4))
        self.assertEqual(0.4205, np.round(d['cont_diff_dist'], decimals=4))

    def test_bhattacharyya_cont_str_input(self):
        var = 'cont_same_dist'
        d = sf.bhattacharyya(self.df_1, self.df_2, continuous_variables=var)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(var in d.index)
        self.assertEqual(0.0138, np.round(d[var], decimals=4))

    def test_bhattacharyya_cont_column_not_in_both(self):
        var = ['cont_same_dist', 'cont_diff_dist', 'cont_not_in_both']
        d = sf.bhattacharyya(self.df_1, self.df_2, continuous_variables=var)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(all([v in d.index for v in var] + [v in var for v in d.index]))
        self.assertTrue(d.isnull()['cont_not_in_both'])

    def test_bhattacharyya_cont_non_numeric_column(self):
        var = ['cont_same_dist', 'cont_diff_dist', 'cont_not_in_both']
        df = self.df_2.assign(cont_not_in_both=np.random.choice(['a','b','c'], size=self.df_2.shape[0]))
        with self.assertRaises(AssertionError) as ctx:
            sf.bhattacharyya(self.df_1, df, continuous_variables=var)
        self.assertEqual('All continuous variables must be numerical', str(ctx.exception))

    # Full tests
    def test_bhattacharyya_full_calculation(self):
        cat = ['cat_same_dist', 'cat_diff_dist']
        cont = ['cont_same_dist', 'cont_diff_dist']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=cat, continuous_variables=cont)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(all([v in d.index for v in cont]
                            + [(v in cat) or (v in cont) for v in d.index]
                            + [v in d.index for v in cat]))
        self.assertEqual(0.0, d['cat_same_dist'])
        self.assertEqual(0.0614, np.round(d['cat_diff_dist'], decimals=4))
        self.assertEqual(0.0138, np.round(d['cont_same_dist'], decimals=4))
        self.assertEqual(0.4205, np.round(d['cont_diff_dist'], decimals=4))

    def test_bhattacharyya_no_column_not_in_both_crossover_cat_to_cont(self):
        cat = ['cat_same_dist', 'cat_diff_dist', 'cat_not_in_both']
        cont = ['cont_same_dist', 'cont_diff_dist']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=cat, continuous_variables=cont)

        self.assertTrue(d.isnull()['cat_not_in_both'])
        self.assertTrue((~d.isnull().drop('cat_not_in_both')).all())

    def test_bhattacharyya_no_column_not_in_both_crossover_cont_to_cat(self):
        cat = ['cat_same_dist', 'cat_diff_dist']
        cont = ['cont_same_dist', 'cont_diff_dist', 'cont_not_in_both']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=cat, continuous_variables=cont)
        self.assertTrue(d.isnull()['cont_not_in_both'])
        self.assertTrue((~d.isnull().drop('cont_not_in_both')).all())
