import pytest
import datetime

import numpy as np
import pandas as pd

from src.train.train import custom_ts_split, get_split_data


class TestCustomTsSplit:

    def test_no_months(self):

        months = np.array([])
        splits = list(custom_ts_split(months))

        assert len(splits) == 0

    def test_split(self):

        months = np.array([
            datetime.datetime(2020, 4, 30),
            datetime.datetime(2020, 5, 31),
            datetime.datetime(2020, 6, 30),
            datetime.datetime(2020, 7, 31),
            datetime.datetime(2020, 8, 31)
        ])

        splits = []

        for i, (start_train, end_train, test_period) in enumerate(custom_ts_split(months, 1)):

            splits.append((start_train, end_train, test_period))

            assert start_train == pd.Timestamp('2020-04-01 00:00:00')
            assert end_train == months[i]
            assert test_period == months[i + 1]

        assert len(splits) == 4

    @pytest.mark.parametrize('train_period', [-2, -1, 0, 5, 6, 10])
    def test_out_of_bounds(self, train_period):

        months = np.array([
            datetime.datetime(2020, 4, 30),
            datetime.datetime(2020, 5, 31),
            datetime.datetime(2020, 6, 30),
            datetime.datetime(2020, 7, 31),
            datetime.datetime(2020, 8, 31)
        ])

        with pytest.raises(ValueError):
            list(custom_ts_split(months, train_period))


class TestGetSplitData:
    def setup_class(self):

        self.start_train = pd.to_datetime('2020-04-30')
        self.end_train = pd.to_datetime('2020-05-31')
        self.test_period = pd.to_datetime('2020-06-30')

    def test_empty_dataframe(self):

        features = pd.DataFrame()

        # AttributeError because dataframe has no required fields: 'user_id', 'month', 'target'
        with pytest.raises(AttributeError):
            X_train, X_test, y_train, y_test = get_split_data(
                features, self.start_train, self.end_train, self.test_period
            )

    def test_split(self):

        features = pd.DataFrame({
            'user_id': [0, 1, 2, 3, 4],
            'month': [
                '2020-04-30',
                '2020-05-31',
                '2020-06-30',
                '2020-07-31',
                '2020-08-31'
            ],
            'feature_1': [0.5, 0, 2.3, -1, 40],
            'feature_2': [-0.5, 0, -2.3, 1, -40],
            'target': [1, 0, 1, 0, 1]
        })

        features['month'] = pd.to_datetime(features['month'])

        X_train, X_test, y_train, y_test = get_split_data(
            features, self.start_train, self.end_train, self.test_period
        )

        assert X_train.columns.tolist() == ['feature_1', 'feature_2']
        assert X_test.columns.tolist() == ['feature_1', 'feature_2']

        assert X_train['feature_1'].tolist() == [0.5, 0]
        assert X_train['feature_2'].tolist() == [-0.5, 0]

        assert X_test['feature_1'].tolist() == [2.3]
        assert X_test['feature_2'].tolist() == [-2.3]

        assert y_train.tolist() == [1, 0]
        assert y_test.tolist() == [1]
