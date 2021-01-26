import pandas as pd
import re


def add_feature31(df):
    """Add some first letter device code"""

    df['feature_31'] = df.feature_21.apply(lambda s: 'None' if s in ['', None] else re.findall(r'[\w]', s)[0])

    return df