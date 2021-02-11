from typing import Text

import pandas as pd


def load_feather(path: Text) -> pd.DataFrame:
    return pd.read_feather(path)
