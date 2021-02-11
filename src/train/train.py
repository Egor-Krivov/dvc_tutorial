import logging
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from typing import Iterator, Tuple

from src.utils.logging import get_logger


def custom_ts_split(months: np.array, train_period: int = 0) -> Iterator:
    """
    Custom datetime splitter.
    Args:
        months {np.array}: months array
        train_period {int}: train period duration, in months
    Yields:
        tuple of start_train, end_train and test_period
    """

    logger = get_logger(__name__, logging.INFO)

    for k, month in enumerate(months):

        start_train = pd.to_datetime(months.min())
        end_train = pd.to_datetime(start_train) + MonthEnd(train_period - 1 + k)
        test_period = pd.to_datetime(end_train + MonthEnd(1))

        if test_period <= pd.to_datetime(months.max()):
            yield start_train, end_train, test_period
        else:
            logger.info(test_period)
            logger.info(months.max())


def get_split_data(features: pd.DataFrame, start_train: pd.Timestamp, end_train: pd.Timestamp, test_period: pd.Timestamp
                   ) -> Tuple[pd.DataFrame]:

    # Get train / test data for the split
    X_train = (features[(features.month >= start_train) & (features.month <= end_train)]
               .drop(columns=['user_id', 'month', 'target'], axis=1))
    X_test = (features[(features.month == test_period)]
              .drop(columns=['user_id', 'month', 'target'], axis=1))
    y_train = features.loc[(features.month >= start_train) & (features.month <= end_train), 'target']
    y_test = features.loc[(features.month == test_period), 'target']

    return X_train, X_test, y_train, y_test
