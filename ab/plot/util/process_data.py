import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use the non-interactive Agg backend

def process_data(df):
    """
    Process raw data to calculate mean and std for metrics, including the `nn` column for models.
    
    Args:
        df (DataFrame): Raw data.
        
    Returns:
        DataFrame: Aggregated statistics.
    """
    # Validate necessary columns
    required_columns = {'task', 'dataset', 'epoch', 'metric', 'accuracy', 'duration', 'nn'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

    # Ensure 'epoch' is numeric for correct sorting
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    if df['epoch'].isna().any():
        raise ValueError("Invalid epoch values detected. Ensure all epochs are numeric.")

    unique_metrics = df['metric'].unique()

    # Initialize an empty list to store aggregated results
    aggregated_frames = []

    # Process each metric separately
    for metric in unique_metrics:
        metric_data = df[df['metric'] == metric]  # Filter data for the current metric
        if metric == 'acc':
            aggregated = metric_data.groupby(['task', 'dataset', 'nn', 'epoch'], as_index=False).agg(
                accuracy_mean=('accuracy', 'mean'),
                accuracy_std=('accuracy', 'std'),
                duration=('duration', 'mean') 
            )
        elif metric == 'iou':
            aggregated = metric_data.groupby(['task', 'dataset', 'nn', 'epoch'], as_index=False).agg(
                iou_mean=('accuracy', 'mean'),  # Assuming 'accuracy' column holds IoU values
                iou_std=('accuracy', 'std'),
                duration=('duration', 'mean') 
            )
        else:
            continue

        # Add the metric column back to the aggregated DataFrame
        aggregated['metric'] = metric
        aggregated_frames.append(aggregated)

    # Combine all aggregated results
    aggregated_data = pd.concat(aggregated_frames, ignore_index=True)

    # Sort by 'epoch' to ensure correct order
    aggregated_data = aggregated_data.sort_values(by=['task', 'dataset', 'nn', 'metric', 'epoch']).reset_index(drop=True)

    return aggregated_data
