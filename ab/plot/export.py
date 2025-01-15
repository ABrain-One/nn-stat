import os

from ab.plot.util.Const import plot_dir, excel_dir

from ab.plot.util.fetch_data import fetch_all_data
from ab.plot.util.filter_raw_data import filter_raw_data
from ab.plot.util.generate_plots import generate_all_plots
from ab.plot.util.process_data import process_data
from ab.plot.xls.export_excel import export_to_excel
from ab.plot.xls.export_raw_excel import export_raw_data_with_plots


# Ensure output directories exist
os.makedirs(excel_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
# os.makedirs("data", exist_ok=True)

# Step 1: Fetch Data
print("Fetching data...")
raw_data = fetch_all_data()

# Step 2: Process Data
print("Processing data...")

aggregated_data = process_data(raw_data)

# If you want to save the raw data and aggregated statistics to CSV files
# raw_data.to_csv("data/raw_data.csv", index=False)
# aggregated_data.to_csv("data/aggregated_statistics.csv", index=False)

# Map 'acc' to 'accuracy'
aggregated_data['metric'] = aggregated_data['metric'].replace({'acc': 'accuracy'})

# Step 3: Generate Plots
print("Generating plots...")
for metric in aggregated_data['metric'].unique():
    metric_data = aggregated_data[aggregated_data['metric'] == metric]
    plot_path = f"{plot_dir}/{metric}_plot.png"
    generate_all_plots(aggregated_data, output_dir=plot_dir)

# # Step 4: Export to Excel
print("Exporting data to Excel...")
export_to_excel(
    aggregated_data=aggregated_data,
    output_file=excel_dir / 'statistics.xlsx',
    plot_dir=plot_dir
)
# Step 5: Filter Raw Data
print("Filtering raw data...")
filtered_raw_data = filter_raw_data(raw_data)
print("Exporting raw data and plots to Excel...")
export_raw_data_with_plots(filtered_raw_data, output_file=excel_dir / 'raw_data.xlsx', plot_dir=plot_dir / 'raw_plots')
