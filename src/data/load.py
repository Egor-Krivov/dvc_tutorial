import pandas as pd
from typing import Text


def load_target(path: Text) -> pd.DataFrame:
    return pd.read_feather(path)


def load_data(path: Text) -> pd.DataFrame:
    return pd.read_feather(path)


def load_features(path: Text) -> pd.DataFrame:
    return pd.read_feather(path)