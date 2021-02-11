"""This script extracts features from raw data."""

import yaml
from typing import Text

import autocommand
import pandas as pd

from src.data.load import load_feather
from src.data.process import process_target, process_data
from src.utils.logging import get_logger


def data_load(config_path: Text) -> None:
    """Load and process data.
    Args:
        config_path {Text}: path to config file
    """
    config = yaml.safe_load(open(config_path))

    log_level = config['base']['log_level']
    target_raw = config['data_load']['target']
    dataset_raw = config['data_load']['dataset']
    target_processed = config['data_load']['target_processed']
    dataset_processed = config['data_load']['dataset_processed']

    logger = get_logger("DATA_LOAD", log_level)

    logger.info('Load dataset')
    target_df = load_feather(target_raw)
    data_df = load_feather(dataset_raw)

    logger.info('Process target')
    target_df = process_target(target_df)

    logger.info('Process dataset')
    data_df = process_data(data_df)
    data_df.dropna(inplace=True)

    logger.info('Save processed data and target')
    target_df.to_feather(target_processed)
    data_df.to_feather(dataset_processed)
    logger.debug(f'Processed data path: {dataset_processed}')
    logger.debug(f'Processed data path: {target_processed}')


autocommand.autocommand(__name__)(data_load)
