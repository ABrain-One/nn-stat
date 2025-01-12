import matplotlib.pyplot as plt

def generate_plot(data, metric, output_path):
    """
    Generate a plot for a specific metric.
    Args:
        data (DataFrame): Aggregated data for the metric.
        metric (str): Metric to plot ('acc' or 'iou').
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data['accuracy_mean'], label='Mean', color='blue')
    plt.fill_between(
        data['epoch'],
        data['accuracy_mean'] - data['accuracy_std'],
        data['accuracy_mean'] + data['accuracy_std'],
        color='blue', alpha=0.2, label='Std Dev'
    )
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.title(f"{data['task'].iloc[0]} - {data['dataset'].iloc[0]} ({metric.upper()})")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
