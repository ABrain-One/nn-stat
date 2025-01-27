import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set global seaborn theme with a visually appealing color palette
sns.set_theme(style="whitegrid")
sns.set_palette("Set2")  # Set a distinct color palette globally

# Helper function to save plots in both PNG and SVG formats
def save_plot(plt_obj, output_name, png_dir, svg_dir):
    """Saves plots in both PNG and SVG formats in their respective directories."""
    png_path = os.path.join(png_dir, f"{output_name}.png")
    svg_path = os.path.join(svg_dir, f"{output_name}.svg")

    # Ensure directories exist
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    # Save files
    plt_obj.savefig(png_path, format="png", dpi=300)  # Save with high DPI for better quality
    plt_obj.savefig(svg_path, format="svg")
    plt_obj.close()
    

# Function to create a consistent color palette
def create_color_palette(unique_values, palette_name="tab20"):
    """Generates a consistent color palette for unique values."""
    palette = sns.color_palette(palette_name, n_colors=max(len(unique_values), 20))
    return {value: color for value, color in zip(unique_values, palette)}


def ensure_grouped_rolling_mean(data):
    """Ensure rolling mean is calculated per group."""
    print("Calculating rolling mean per group...")
    data['rolling_mean'] = data.groupby(['task', 'dataset', 'nn'])['accuracy_mean'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    return data

# Function to set x-axis ticks dynamically based on epoch range
def set_epoch_ticks(ax, max_epoch):
    ticks = np.arange(0, max_epoch + 1, 10)
    ax.set_xticks(ticks)


# Function to plot rolling mean
def plot_rolling_mean(data, metric, color_map, max_models=10):
    metric_column = f'{metric}_mean'
    data['rolling_mean'] = data[metric_column].rolling(window=5, min_periods=1).mean()
    unique_models = data['nn'].unique()

    # Limit the number of models displayed
    if len(unique_models) > max_models:
        print(f"Limiting models displayed to {max_models} for clarity.")
        unique_models = unique_models[:max_models]

    plt.figure(figsize=(20, 12))  # Increased figure size for clarity

    for idx, model in enumerate(unique_models):
        model_data = data[data['nn'] == model]
        if model not in color_map:
            color_map[model] = 'gray'

        color = color_map[model]
        linestyle = '-' if idx % 2 == 0 else '--'  
        plt.plot(
            model_data['epoch'],
            model_data[metric_column],
            label=f"{model} - Mean",
            color=color,
            marker='o',
            linewidth=1.5,
            linestyle=linestyle
        )
        plt.plot(
            model_data['epoch'],
            model_data['rolling_mean'],
            label=f"{model} - Rolling Mean",
            color=color,
            linestyle=':',
            linewidth=2
        )

    ax = plt.gca()
    set_epoch_ticks(ax, data['epoch'].max())  # Adjust x-axis ticks dynamically
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(metric.capitalize(), fontsize=16)
    plt.title(f"{data['task'].iloc[0]} - {data['dataset'].iloc[0]} (Rolling Mean)", fontsize=18, fontweight="bold")
    plt.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, title="Neural Network Model")
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()



# Function to plot mean and standard deviation
def plot_mean_std(data, metric, color_map):
    metric_columns = {
        'accuracy': ('accuracy_mean', 'accuracy_std'),
        'iou': ('iou_mean', 'iou_std')
    }

    if metric not in metric_columns:
        raise ValueError(f"Unsupported metric '{metric}'.")

    mean_col, std_col = metric_columns[metric]
    unique_models = data['nn'].unique()
    plt.figure(figsize=(14, 8))  # Increased figure size

    for model in unique_models:
        model_data = data[data['nn'] == model]
        if model not in color_map:
            color_map[model] = 'gray'

        color = color_map[model]
        plt.plot(
            model_data['epoch'],
            model_data[mean_col],
            label=f"{model} - Mean",
            color=color,
            marker='o',
            linewidth=2
        )
        plt.fill_between(
            model_data['epoch'],
            model_data[mean_col] - model_data[std_col],
            model_data[mean_col] + model_data[std_col],
            color=color,
            alpha=0.3
        )

    ax = plt.gca()
    set_epoch_ticks(ax, data['epoch'].max())  # Adjust x-axis ticks
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.title(f"{data['task'].iloc[0]} - {data['dataset'].iloc[0]} ({metric.capitalize()})", fontsize=16, fontweight="bold")
    plt.legend(fontsize=10, loc="best", bbox_to_anchor=(1.05, 1))  # Adjust legend
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()



def plot_box(data, metric):
    plt.figure(figsize=(14, 8))  # Adjust figure size
    sns.boxplot(
        x='epoch',
        y=metric,
        data=data,
        width=0.7  # Adjust box width
    )

    ax = plt.gca()

    # Adjust x-axis ticks
    max_epoch = data['epoch'].max()
    tick_values = list(range(0, max_epoch + 10, 10))  # Ensure ticks are in steps of 10
    ax.set_xticks(tick_values)  # Set custom ticks
    ax.set_xticklabels(tick_values)  # Adjust labels to match ticks

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.title(f"{data['task'].iloc[0]} - {data['dataset'].iloc[0]} ({metric.capitalize()} Distribution)", fontsize=16, fontweight="bold")
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()


# Function to plot correlation heatmap
def plot_correlation_heatmap(data):
    correlation_data = data[['accuracy_mean', 'accuracy_std', 'iou_mean', 'iou_std']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_data,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        cbar_kws={'label': 'Correlation'},
        annot_kws={"size": 10}  # Smaller annotations for clarity
    )
    plt.title("Metric Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()

# Function to plot scatter plot of training time vs metrics
def plot_metric_vs_time(data, metric, task_color_map, model_color_map):
    """
    Plots Training Time vs Accuracy (or IoU depending on the metric).
    The x-axis represents Training Time, and the y-axis represents the selected metric.
    """
    plt.figure(figsize=(10, 6))

    # Determine the hue column and corresponding color map
    hue_col = 'task' if 'task' in data.columns else 'nn'
    color_map = task_color_map if hue_col == 'task' else model_color_map

    # Handle missing keys in color_map
    missing_keys = set(data[hue_col].unique()) - set(color_map.keys())
    for key in missing_keys:
        print(f"Warning: '{key}' is missing from color_map. Adding a default color.")
        color_map[key] = 'gray'

    # Create scatter plot
    sns.scatterplot(
        x='duration',  # Training Time on the x-axis
        y=f'{metric}_mean',  # Metric (e.g., accuracy) on the y-axis
        hue=hue_col,
        palette=color_map,
        data=data,
        s=100,  # Marker size
        alpha=0.8
    )

    # Customizing plot aesthetics
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Training Time (nanoseconds)", fontsize=14)
    plt.ylabel(f"{metric.capitalize()}", fontsize=14)
    plt.title(f"{metric.capitalize()} vs Training Time", fontsize=16, fontweight="bold")
    plt.legend(title="Neural Network Model" if hue_col == 'nn' else "Task", fontsize=12, loc="best")
    plt.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()

#  distribution of duration values for the first epoch
def plot_model_duration_distribution_with_annotations(data, output_name, png_dir, svg_dir):
    """Generates grouped distribution plots of model durations for the first epoch with annotations."""
    first_epoch_data = data[data['epoch'] == 1]

    for task in first_epoch_data["task"].unique():
        task_data = first_epoch_data[first_epoch_data["task"] == task]

        plt.figure(figsize=(18, 10))

        # Violin plot for distribution
        sns.violinplot(
            x="dataset",
            y="duration",
            data=task_data,
            split=True,
            density_norm="width",  # Normalize by width
            inner="quartile",
            color="skyblue",  # Uniform color for simplicity
        )

        # Swarm plot for individual points
        sns.swarmplot(
            x="dataset",
            y="duration",
            data=task_data,
            dodge=True,
            size=6,
            alpha=0.8,
            color="orange",  # Set yellow dots
        )

        # Add title and labels
        plt.title(f"Model Duration Distribution ({task})", fontsize=18, fontweight="bold")
        plt.xlabel("Dataset", fontsize=14)
        plt.ylabel("Training Time (nanoseconds)", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yscale("log")  # Logarithmic scale for duration
        plt.grid(linestyle="--", alpha=0.5)

        # Add legend for explanation
        plt.annotate(
            "Yellow dots: Individual models\nViolin plot: Distribution of durations",
            xy=(0.02, 0.9),  # Adjust location
            xycoords="axes fraction",
            fontsize=12,
            color="black",
            backgroundcolor="white",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7),
        )

        plt.tight_layout()
        task_output_name = f"{output_name}_{task.replace(' ', '_')}"
        save_plot(plt, task_output_name, png_dir, svg_dir)


# Main function to generate all plots
def generate_all_plots(data, png_dir, svg_dir):
    metrics = ['accuracy', 'iou']

    # Generate a global color map for tasks
    unique_tasks = data['task'].unique()
    task_color_map = create_color_palette(unique_tasks)


    # Ensure rolling mean is calculated per group
    data = ensure_grouped_rolling_mean(data)

    for metric in metrics:
        # Filter data by metric
        metric_data = data[data['metric'] == metric]

        if metric_data.empty:
            print(f"No data available for metric: {metric}")
            continue

        for (task, dataset), group_data in metric_data.groupby(['task', 'dataset']):

            if group_data.empty:
                print(f"No data found for {task}, {dataset}, {metric}")
                continue

            # Dynamically generate a color_map for models within the group
            unique_models = group_data['nn'].unique()
            model_color_map = create_color_palette(unique_models)

            # Save Mean and Std Plot
            mean_std_output_name = f"{task}_{dataset}_{metric}_mean_std"
            try:
                plot_mean_std(group_data, metric, model_color_map)
                save_plot(plt, mean_std_output_name, png_dir, svg_dir)
            except ValueError as e:
                print(f"Error plotting mean and std for {metric}: {e}")

            # Save Box Plot
            box_output_name = f"{task}_{dataset}_{metric}_box"
            try:
                plot_box(group_data, f"{metric}_mean")
                save_plot(plt, box_output_name, png_dir, svg_dir)
            except ValueError as e:
                print(f"Error plotting box plot for {metric}: {e}")

            # Save Rolling Mean Plot
            rolling_mean_output_name = f"{task}_{dataset}_{metric}_rolling_mean"
            try:
                plot_rolling_mean(group_data, metric, model_color_map)
                save_plot(plt, rolling_mean_output_name, png_dir, svg_dir)
            except ValueError as e:
                print(f"Error plotting rolling mean for {metric}: {e}")

        # Training Time vs Metrics
        time_vs_metric_output_name = f"{metric}_vs_training_time"
        try:
            plot_metric_vs_time(
                metric_data,
                metric,
                task_color_map,  # Task-level coloring
                model_color_map  # Model-level coloring
            )
            save_plot(plt, time_vs_metric_output_name, png_dir, svg_dir)
        except ValueError as e:
            print(f"Error plotting training time vs {metric}: {e}")

    # Correlation Heatmap
    if not data[['accuracy_mean', 'accuracy_std', 'iou_mean', 'iou_std']].isna().all().all():
        heatmap_output_name = "correlation_heatmap"
        try:
            plot_correlation_heatmap(data)
            save_plot(plt, heatmap_output_name, png_dir, svg_dir)
        except ValueError as e:
            print(f"Error plotting correlation heatmap: {e}")

    # First Epoch Duration Distribution
    first_epoch_distribution_output_name = "first_epoch_duration_distribution"
    try:
        plot_model_duration_distribution_with_annotations(data, first_epoch_distribution_output_name, png_dir, svg_dir)
    except ValueError as e:
        print(f"Error plotting duration distribution: {e}")

