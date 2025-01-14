import pandas as pd

def filter_raw_data(raw_data):
    """
    Filter raw data to retain only relevant columns.
    Args:
        raw_data (DataFrame): The raw data fetched from the database.
    Returns:
        DataFrame: Filtered raw data containing only relevant columns.
    """
    # Define the relevant columns to keep
    relevant_columns = ['task', 'dataset', 'metric', 'epoch', 'duration', 'accuracy', 'nn']

    # Filter the data
    filtered_data = raw_data[relevant_columns]

    return filtered_data
