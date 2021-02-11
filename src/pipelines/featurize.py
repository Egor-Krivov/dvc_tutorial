"""This script extracts features from raw data."""

import yaml
from typing import Text

import autocommand
import pandas as pd

from src.data.load import load_feather
from src.data.process import process_target, process_features
from src.utils.logging import get_logger


def featurize(config_path: Text) -> None:
    """Extract features and save it.
    Args:
        config_path {Text}: path to config file
    """
    config = yaml.safe_load(open(config_path))

    log_level = config['base']['log_level']
    target_path = config['data_load']['target_processed']
    dataset_path = config['data_load']['dataset_processed']
    features_path = config['featurize']['features_path']

    logger = get_logger("FEATURIZE", log_level)

    logger.info('Load dataset')
    target_df = load_feather(target_path)
    features_df = load_feather(dataset_path)

    logger.info('Process dataset')
    features_df = process_features(features_df, target_df)

    logger.info('Add target column')
    features = features_df.copy()
    features = pd.merge(
        left=features,
        right=target_df,
        how='left',
        on=['user_id', 'month']
    )

    logger.info('Process nulls')
    features.dropna(inplace=True)

    logger.info('Save features')
    features.to_feather(features_path)
    logger.debug(f'Features path: {features_path}')


autocommand.autocommand(__name__)(featurize)
