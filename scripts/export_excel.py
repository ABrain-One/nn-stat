from openpyxl import Workbook
from openpyxl.drawing.image import Image
import os


def export_to_excel(aggregated_data, output_file, plot_dir, image_width=400, image_height=300):
    """
    Export aggregated data and categorized plots to an Excel file.
    Args:
        aggregated_data (DataFrame): Aggregated statistics.
        output_file (str): Path to save the Excel file.
        plot_dir (str): Directory containing plots.
        image_width (int): Width of embedded images (default: 400).
        image_height (int): Height of embedded images (default: 300).
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Statistics"

    # Add headers
    ws.append(aggregated_data.columns.tolist())

    # Add data rows
    for row in aggregated_data.itertuples(index=False):
        ws.append(row)

    # Define plot categories and their corresponding file patterns
    plot_categories = {
        "Mean and Std Dev": "{task}_{dataset}_{metric}_mean_std.png",
        "Box Plot": "{task}_{dataset}_{metric}_box.png",
        "Rolling Mean": "{task}_{dataset}_{metric}_rolling_mean.png",
        "Task Comparison (Accuracy)": "task_comparison_accuracy.png",
        "Task Comparison (IoU)": "task_comparison_iou.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Time vs Accuracy": "time_vs_accuracy.png",
        "Time vs IoU": "time_vs_iou.png",
    }

    # Start embedding images below the data
    start_row = len(aggregated_data) + 5  # Leave some space after the data
    for category, file_suffix in plot_categories.items():
        # Add a category header
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=8)
        ws.cell(start_row, 1, category)
        start_row += 2  # Leave a gap for better readability

        # Processing Mean and Std Dev, Box Plot, and Rolling Mean plots (parallel layout)
        if category in ["Mean and Std Dev", "Box Plot", "Rolling Mean"]:
            col_offset = 1  # Start placing images from the first column
            row_start = start_row  # Track the starting row for this category
            max_row_used = row_start  # Track the maximum row used to adjust spacing
            for metric in aggregated_data['metric'].unique():
                for (task, dataset) in aggregated_data.groupby(['task', 'dataset']).groups.keys():
                    plot_file = file_suffix.format(task=task, dataset=dataset, metric=metric).replace(" ", "_")
                    plot_path = os.path.join(plot_dir, plot_file)

                    if os.path.exists(plot_path):
                        img = Image(plot_path)
                        img.width = image_width
                        img.height = image_height
                        img.anchor = f"{chr(64 + col_offset)}{row_start}"  # Place image horizontally
                        ws.add_image(img)
                        col_offset += 6  # Increased column spacing

                        # Reset to the first column if the row fills
                        if col_offset > 12:  # Adjust for maximum columns before wrapping
                            col_offset = 1
                            row_start += 20  # Adjust row spacing for resized images

                        max_row_used = max(max_row_used, row_start)  # Update maximum row used
                    else:
                        print(f"File not found: {plot_path}")

            # Update the start_row for the next category
            start_row = max_row_used + 2  # Add a small gap between categories

        else:
            # General category plots (sequential layout)
            plot_path = os.path.join(plot_dir, file_suffix)

            # Debug: Log each general file path
            print(f"Looking for file: {plot_path}")

            if os.path.exists(plot_path):
                img = Image(plot_path)
                img.width = image_width
                img.height = image_height
                img.anchor = f"A{start_row}"  # Place image vertically
                ws.add_image(img)
                start_row += 20  # Adjust row spacing for resized images
            else:
                print(f"File not found: {plot_path}")


    # Save the Excel file
    wb.save(output_file)
    print(f"Excel file saved to {output_file}")
