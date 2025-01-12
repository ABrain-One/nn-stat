import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use the non-interactive Agg backend
# import matplotlib.pyplot as plt


def process_data(df):
    """
    Process raw data to calculate mean and std for metrics.
    Args:
        df (DataFrame): Raw data.
    Returns:
        DataFrame: Aggregated statistics.
    """
    # Validate necessary columns
    required_columns = {'task', 'dataset', 'epoch', 'metric', 'accuracy'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

    # Group by task, dataset, epoch, and metric, then aggregate
    aggregated_data = df.groupby(['task', 'dataset', 'epoch', 'metric'], as_index=False).agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std')
    )
    return aggregated_data


if __name__ == "__main__":
    # Example usage
    raw_data = pd.read_csv("../data/raw_data.csv")
    aggregated_data = process_data(raw_data)
    print(aggregated_data.head())
    aggregated_data.to_csv("../data/aggregated_data.csv", index=False)
