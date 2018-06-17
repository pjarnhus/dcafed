import unittest
import pandas as pd
import numpy as np
from dcafed import stat_func as sf


class StatFuncTests(unittest.TestCase):
    # executed prior to each test
    def setUp(self):
        np.random.seed(42)
        s = 100
        self.df_1 = pd.DataFrame({'cat_1': [0, 1] * int(s/2),
                                  'cat_2': [0] * int(s/3) + [1] * (s - int(s / 3)),
                                  'cat_3': [0] * int(s / 3) + [1] * int(s / 3) + [2] * (s - 2*int(s/3)),
                                  'cont_1': np.random.randn(s),
                                  'cont_2': np.random.randn(s) + 1,
                                  'cont_3': np.random.randn(s)})

        self.df_2 = pd.DataFrame({'cat_1': [0, 1] * int(s / 2),
                                  'cat_2': [1] * int(s / 3) + [0] * (s - int(s / 3)),
                                  'cat_3': [0, 1] * int(s / 2),
                                  'cat_4': np.random.randint(50, size=s),
                                  'cont_1': np.random.randn(s),
                                  'cont_2': np.random.randn(s) - 1})

        self.str_1 = 'abc'
        self.str_2 = 'def'

    # executed after every test
    def tearDown(self):
        pass

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

    def test_bhattacharyya_cat_distance(self):
        var = ['cat_1', 'cat_2']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=var)
        self.assertTrue(isinstance(d, pd.Series))
        self.assertTrue(all([v in d.index for v in var]+[v in var for v in d.index]))
        self.assertEqual(0.0, d['cat_1'])
        self.assertEqual(0.0614, np.round(d['cat_2'], decimals=4))

    def test_bhattacharyya_cat_value_mismatch(self):
        var = ['cat_1', 'cat_2', 'cat_3']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=var)
        self.assertTrue(d.isnull()['cat_3'])

    def test_bhattacharyya_cat_str_input(self):
        var = 'cat_1'
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=var)
        self.assertEqual(0.0, d[var])

    def test_bhattacharyya_cat_non_matching_column(self):
        var = ['cat_1', 'cat_2', 'cat_4']
        d = sf.bhattacharyya(self.df_1, self.df_2, categorical_variables=var)
        self.assertTrue(d.isnull()['cat_4'])
