from openpyxl import Workbook
from openpyxl.drawing.image import Image
import os

def export_to_excel(aggregated_data, output_file, plot_dir):
    """
    Export aggregated data and plots to an Excel file.
    Args:
        aggregated_data (DataFrame): Aggregated statistics.
        output_file (str): Path to save the Excel file.
        plot_dir (str): Directory containing plots.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Statistics"

    # Add headers
    ws.append(aggregated_data.columns.tolist())

    # Add data rows
    for row in aggregated_data.itertuples(index=False):
        ws.append(row)

    # Embed plots
        # Embed plots
    # for metric in aggregated_data['metric'].unique():
    #     plot_path = os.path.join(plot_dir, f"{metric}_plot.png")
    #     if os.path.exists(plot_path):
    #         img = Image(plot_path)
    #         ws.add_image(img, "I2")
    row_offset = 2  # Starting row for images
    for idx, metric in enumerate(aggregated_data['metric'].unique()):
        plot_path = os.path.join(plot_dir, f"{metric}_plot.png")
        if os.path.exists(plot_path):
            img = Image(plot_path)
            img.anchor = f"I{row_offset}"  # Adjust row position for each image
            ws.add_image(img)
            row_offset += 15  # Move to the next row for the next image

    wb.save(output_file)
    print(f"Excel file saved to {output_file}")
