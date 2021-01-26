import pandas as pd

def process_features(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    features_df = features_df.loc[features_df.user_id.isin(target_df.user_id),]
    features_df['month'] = pd.to_datetime(features_df['month'])

    return features_df


def process_target(target_df: pd.DataFrame) -> pd.DataFrame:
    target_df['month'] = pd.to_datetime(target_df['month'])

    return target_df


def process_data(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df['month'] = pd.to_datetime(data_df['month'])

    return data_df