import pandas as pd
import os

def load_data(file_path):
    """
    this function takes input the file path of data and returns a pandas dataframe
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no such file or directory on {file_path}")
    else:
        return pd.read_csv(file_path)