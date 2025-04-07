import polars as pl
from pathlib import Path
from typing import Union, IO

def read_tabular_file(file: Union[str, Path, IO]) -> pl.DataFrame:
    """
    Reads a CSV or TSV file into a Polars DataFrame.
    Accepts both file paths and file-like objects.
    """
    
    if hasattr(file, 'filename'):
        filename = file.filename
    else:
        filename = str(file)

    if filename.lower().endswith(".tsv"):
        return pl.read_csv(file, separator="\t")
    else:  # Default to CSV
        return pl.read_csv(file)
