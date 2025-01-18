import os

import matplotlib.pyplot as plt
import seaborn as sns

from ab.plot.util.Const import plot_dir

sns.set_theme(style="whitegrid")

# Function for line plots with mean and std deviation
def plot_mean_std(data, metric, output_path):
    metric_columns = {
        'accuracy': ('accuracy_mean', 'accuracy_std'),
        'iou': ('iou_mean', 'iou_std')
    }

    if metric not in metric_columns:
        raise ValueError(f"Unsupported metric '{metric}'.")

    mean_col, std_col = metric_columns[metric]

    plt.figure(figsize=(12, 6))
    plt.plot(data['epoch'], data[mean_col], label='Mean', color='blue')
    plt.fill_between(
        data['epoch'],
        data[mean_col] - data[std_col],
        data[mean_col] + data[std_col],
        color='blue', alpha=0.2, label='Std Dev'
    )

    # Customize x-axis ticks and spacing
    unique_epochs = sorted(data['epoch'].unique())
    plt.xticks(unique_epochs, rotation=0)  # Ensure all epochs are shown
    plt.tick_params(axis='x', which='major', labelsize=10)

    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"{data['task'].iloc[0]} - {data['dataset'].iloc[0]} ({metric.capitalize()})")
    plt.legend()
    plt.grid()
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(output_path)
    plt.close()


# Function for box plots
def plot_box(data, metric, output_path):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='epoch', y=metric, data=data)

    # Customize x-axis ticks and spacing
    unique_epochs = sorted(data['epoch'].unique())
    plt.xticks(unique_epochs, rotation=0)  # Ensure all epochs are shown
    plt.tick_params(axis='x', which='major', labelsize=10)

    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"{data['task'].iloc[0]} - {data['dataset'].iloc[0]} ({metric.capitalize()} Distribution)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Function for heatmap of metric correlations
def plot_correlation_heatmap(data, output_path):
    correlation_data = data[['accuracy_mean', 'accuracy_std', 'iou_mean', 'iou_std']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_data, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Metric Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()

# Function for scatter plot: training time vs metrics
def plot_time_vs_metric(data, metric, output_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration', y=f'{metric}_mean', hue='task', style='dataset', data=data)
    plt.xlabel("Training Time (nanposeconds)")
    plt.ylabel(metric.capitalize())
    plt.title(f"Training Time vs {metric.capitalize()}")
    plt.legend(title="Task/Dataset")
    plt.grid()
    plt.savefig(output_path)
    plt.close()

# Function to plot rolling mean
def plot_rolling_mean(data, metric, output_path):
    metric_column = f'{metric}_mean'

    data['rolling_mean'] = data[metric_column].rolling(window=5, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(data['epoch'], data[metric_column], label='Mean', color='blue')
    plt.plot(data['epoch'], data['rolling_mean'], label='Rolling Mean', color='orange', linestyle='--')

    # Customize x-axis ticks and spacing
    unique_epochs = sorted(data['epoch'].unique())
    plt.xticks(unique_epochs, rotation=0)  # Ensure all epochs are shown
    plt.tick_params(axis='x', which='major', labelsize=10)

    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"{data['task'].iloc[0]} - {data['dataset'].iloc[0]} (Rolling Mean)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Main function to generate all plots
def generate_all_plots(data, output_dir=plot_dir):
    metrics = ['accuracy', 'iou']
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for metric in metrics:
        for (task, dataset), group_data in data.groupby(['task', 'dataset']):
            print(f"Processing task: {task}, dataset: {dataset}, metric: {metric}")

            metric_data = group_data[group_data['metric'] == metric]

            # Skip if no data for the metric
            if metric_data.empty:
                print(f"No data found for {task}, {dataset}, {metric}")
                continue

            # Generate mean and std deviation plot
            mean_std_path = f"{output_dir}/{task}_{dataset}_{metric}_mean_std.png"
            print(f"Generating Mean & Std Dev plot: {mean_std_path}")
            plot_mean_std(metric_data, metric, mean_std_path)

            # Generate box plot
            box_path = f"{output_dir}/{task}_{dataset}_{metric}_box.png"
            print(f"Generating Box Plot: {box_path}")
            plot_box(metric_data, f'{metric}_mean', box_path)

            # Generate rolling mean plot
            rolling_mean_path = f"{output_dir}/{task}_{dataset}_{metric}_rolling_mean.png"
            print(f"Generating Rolling Mean plot: {rolling_mean_path}")
            plot_rolling_mean(metric_data, metric, rolling_mean_path)

    # Generate correlation heatmap
    heatmap_path = f"{output_dir}/correlation_heatmap.png"
    print(f"Generating Correlation Heatmap: {heatmap_path}")
    plot_correlation_heatmap(data, heatmap_path)

    # Generate scatter plot for training time vs metrics
    time_vs_accuracy_path = f"{output_dir}/time_vs_accuracy.png"
    print(f"Generating Time vs Accuracy plot: {time_vs_accuracy_path}")
    plot_time_vs_metric(data, 'accuracy', time_vs_accuracy_path)

    time_vs_iou_path = f"{output_dir}/time_vs_iou.png"
    print(f"Generating Time vs IoU plot: {time_vs_iou_path}")
    plot_time_vs_metric(data, 'iou', time_vs_iou_path)
