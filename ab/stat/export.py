import os

from ab.stat.util.Const import excel_dir, png_dir_stat, svg_dir_stat, png_dir_raw, svg_dir_raw, raw_xlsx, stat_xlsx

from ab.stat.util.fetch_data import fetch_all_data
from ab.stat.util.filter_raw_data import filter_raw_data
from ab.stat.util.generate_plots import generate_all_plots
from ab.stat.util.process_data import process_data
from ab.stat.xls.export_excel import export_to_excel
from ab.stat.xls.export_raw_excel import export_raw_data_with_plots


def main():
    # Ensure output directories exist
    os.makedirs(png_dir_stat, exist_ok=True)
    os.makedirs(svg_dir_stat, exist_ok=True)
    os.makedirs(png_dir_raw, exist_ok=True)
    os.makedirs(svg_dir_raw, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)

    # Step 1: Fetch Data
    print("Fetching data...")
    raw_data = fetch_all_data()

    # Step 2: Process Data
    print("Processing data...")

    aggregated_data = process_data(raw_data)

    # Map 'acc' to 'accuracy'
    aggregated_data['metric'] = aggregated_data['metric'].replace({'acc': 'accuracy'})

    print("Generating plots...")
    for metric in aggregated_data['metric'].unique():
        metric_data = aggregated_data[aggregated_data['metric'] == metric]
        png_path = png_dir_stat / f"{metric}_plot.png"
        svg_path = svg_dir_stat / f"{metric}_plot.svg"
        generate_all_plots(aggregated_data, png_dir=png_dir_stat, svg_dir=svg_dir_stat)

    # Step 4: Export to Excel
    print("Exporting data to Excel...")
    export_to_excel(
        aggregated_data=aggregated_data,
        output_file=stat_xlsx,
        plot_dir=png_dir_stat
    )
    # Step 5: Filter Raw Data
    print("Filtering raw data...")
    filtered_raw_data = filter_raw_data(raw_data)
    print("Exporting raw data and plots to Excel...")

    # Step 6: Export Raw Data and Plots to Excel
    export_raw_data_with_plots(
        filtered_raw_data,
        output_file=raw_xlsx,
        png_dir=png_dir_raw,
        svg_dir=svg_dir_raw)


if __name__ == "__main__":
    main()
