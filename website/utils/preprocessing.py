import polars as pl
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from typing import Tuple, Dict


def preprocess_polars_for_ml(df: pl.DataFrame) -> pl.DataFrame:
    """
    Minimal preprocessing: encode all textual columns as categorical,
    and optionally convert booleans to ints.
    """
    processed = df.clone()

    # Cast string columns to categorical
    string_cols = [col for col, dtype in zip(processed.columns, processed.dtypes) if dtype == pl.Utf8]
    processed = processed.with_columns([
        pl.col(col).cast(pl.Categorical) for col in string_cols
    ])

    # Convert boolean columns to integers (if any)
    bool_cols = [col for col, dtype in zip(processed.columns, processed.dtypes) if dtype == pl.Boolean]
    processed = processed.with_columns([
        pl.col(col).cast(pl.Int8) for col in bool_cols
    ])

    return processed

def label_encode_polars_categoricals(df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encodes all categorical columns in a Polars DataFrame.
    Returns a new DataFrame and a dict of LabelEncoders for inverse transform.
    """
    # Convert to pandas
    df_pd = df.to_pandas()

    le_dict = {}
    for col in df.columns:
        if df.schema[col] == pl.Categorical:
            le = LabelEncoder()
            df_pd[col] = le.fit_transform(df_pd[col])
            le_dict[col] = le

    # Convert back to Polars
    df_encoded = pl.from_pandas(df_pd)

    return df_encoded, le_dict
