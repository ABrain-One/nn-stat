from scripts.fetch_data import fetch_all_data
from scripts.process_data import process_data
from scripts.generate_plots import  generate_all_plots
from scripts.export_excel import export_to_excel
from scripts.filter_raw_data import filter_raw_data
import os

# Ensure output directories exist
os.makedirs("excel", exist_ok=True)
os.makedirs("plots", exist_ok=True)
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
    plot_path = f"plots/{metric}_plot.png"
    generate_all_plots(aggregated_data, output_dir="./plots")

# # Step 4: Export to Excel
print("Exporting data to Excel...")
export_to_excel(
    aggregated_data=aggregated_data,
    output_file="excel/statistics.xlsx",
    plot_dir="plots"
)
# Step 5: Filter Raw Data
print("Filtering raw data...")
filtered_raw_data = filter_raw_data(raw_data)
filtered_raw_data.to_excel("excel/raw_data.xlsx", index=False)
