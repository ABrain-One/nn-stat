from ab.nn.api import data
import pandas as pd

def fetch_all_data():
    """
    Fetch all data from the database.
    Returns:
        DataFrame: Raw data from the database.
    """
    return data()

if __name__ == "__main__":
    df = fetch_all_data()
    print(df.head())
