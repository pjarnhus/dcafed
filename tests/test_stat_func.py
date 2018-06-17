import unittest
import pandas as pd
import numpy as np
from dcafed import stat_func as sf


class StatFuncTests(unittest.TestCase):
    # executed prior to each test
    def setUp(self):
        np.random.seed(42)
        self.df_1 = pd.DataFrame({'cat_1': np.random.randint(20, size=1000),
                                  'cat_2': np.random.randint(100, size=1000),
                                  'cont_1': np.random.randn(1000),
                                  'cont_2': np.random.randn(1000)+1,
                                  'cont_3': np.random.randn(1000)})

        self.df_2 = pd.DataFrame({'cat_1': np.random.randint(20, size=1000),
                                  'cat_2': np.random.randint(80, size=1000),
                                  'cat_3': np.random.randint(50, size=1000),
                                  'cont_1': np.random.randn(1000),
                                  'cont_2': np.random.randn(1000)-1})

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
