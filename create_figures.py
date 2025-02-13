import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_file_path = 'VideoResultsNEW.xlsx'

# Load the data (handling multiple sheets)
data = pd.read_excel(data_file_path, sheet_name=None)

# Create a dictionary to store dataframes for each sheet
dataframes = {}

# Iterate through each sheet and store the dataframe
for sheet_name, df in data.items():
    if sheet_name not in ['Accuracy', 'Performance']:
        continue
    dataframes[sheet_name] = df

# Plot for Accuracy results
if 'Accuracy' not in dataframes:
    print("Error: 'Accuracy' sheet not found in the Excel file.")
else:
    df_accuracy = dataframes['Accuracy']
    
    # Verify that the expected columns exist
    for col in ['Model', 'Central Unit', 'Marker Opacity']:
        if col not in df_accuracy.columns:
            print(f"Warning: Expected column '{col}' is missing.")
    
    # Assume that the last 4 columns are the performance metrics (percentage values)
    perf_metrics = df_accuracy.columns[-4:]
    
    # Compute the average performance across the 4 performance metric columns for each record
    df_accuracy['Performance'] = df_accuracy[perf_metrics].mean(axis=1)
    
    # Group by 'Model' and 'Marker Opacity' to aggregate data across the central units
    aggregated = df_accuracy.groupby(['Model', 'Marker Opacity'])['Performance'].agg(['mean', 'std']).reset_index()
    
    # Sort the data for consistent plotting order
    aggregated = aggregated.sort_values(by=['Model', 'Marker Opacity'])
    
    # Increase font sizes for all elements on this plot
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16
    })
    
    # Plot a bar chart with bars color-coded by Marker Opacity and x-axis labeled by Model
    plt.figure(figsize=(14, 6))
    unique_models = aggregated['Model'].unique()
    unique_opacities = aggregated['Marker Opacity'].unique()
    colors = sns.color_palette("coolwarm", len(unique_opacities))
    opacity_color_map = dict(zip(unique_opacities, colors))
    
    bar_width = 0.15
    gap_between_models = 0.05  # Reduce the gap between model groups
    for i, model in enumerate(unique_models):
        for j, opacity in enumerate(unique_opacities):
            subset = aggregated[(aggregated['Model'] == model) & (aggregated['Marker Opacity'] == opacity)]
            x_position = i * (bar_width * len(unique_opacities) + gap_between_models) + j * bar_width
            plt.bar(x_position, subset['mean'].values[0], yerr=subset['std'].values[0], capsize=10,
                    color=opacity_color_map[opacity], width=bar_width, label=f'Opacity {opacity}' if i == 0 else "")
    
    plt.xticks([i * (bar_width * len(unique_opacities) + gap_between_models) + (len(unique_opacities) - 1) * bar_width / 2 for i in range(len(unique_models))], unique_models, fontsize=20)
    plt.xlabel("Model", fontsize=24)
    plt.ylabel("Performance (%)", fontsize=24)
    plt.legend(title='Marker Opacity', title_fontsize=20, fontsize=16)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Pre-recorded Accuracy.png")
    plt.show()

# Plot for Performance results (using a grouped bar chart aggregated across marker opacity levels)
if 'Performance' not in dataframes:
    print("Error: 'Performance' sheet not found in the Excel file.")
else:
    df_performance = dataframes['Performance']
    
    # Verify that the expected columns exist
    expected_columns = ['Model', 'Central Unit', 'Marker Opacity', 'Avg FPS', 'Std FPS']
    missing_columns = [col for col in expected_columns if col not in df_performance.columns]
    if missing_columns:
        print(df_performance.columns)
        for col in missing_columns:
            print(f"Warning: Expected column '{col}' is missing.")
    
    # Group by 'Model' and 'Central Unit' to aggregate data across the marker opacity levels
    aggregated_perf = df_performance.groupby(['Model', 'Central Unit'])[['Avg FPS', 'Std FPS']].mean().reset_index()
    
    # Sort the data for consistent plotting order
    aggregated_perf = aggregated_perf.sort_values(by=['Model', 'Central Unit'])
    
    # Define unique models and central units, and create a color mapping based on Central Unit
    unique_models = aggregated_perf['Model'].unique()
    unique_central_units = aggregated_perf['Central Unit'].unique()
    colors = sns.color_palette("coolwarm", len(unique_central_units))
    central_color_map = dict(zip(unique_central_units, colors))
    
    bar_width = 0.15
    gap_between_models = 0.05
    plt.figure(figsize=(14, 6))
    
    # Plot grouped bars for each Model and Central Unit combination with error bars
    for i, model in enumerate(unique_models):
        for j, central in enumerate(unique_central_units):
            subset = aggregated_perf[(aggregated_perf['Model'] == model) & (aggregated_perf['Central Unit'] == central)]
            if not subset.empty:
                x_position = i * (bar_width * len(unique_central_units) + gap_between_models) + j * bar_width
                plt.bar(x_position, subset['Avg FPS'].values[0], yerr=subset['Std FPS'].values[0], capsize=10,
                        color=central_color_map[central], width=bar_width, label=f'Central Unit {central}' if i == 0 else "")
    
    # Set the x-ticks to be at the center of each Model group
    xtick_positions = [i * (bar_width * len(unique_central_units) + gap_between_models) + ((len(unique_central_units) - 1) * bar_width) / 2 for i in range(len(unique_models))]
    plt.xticks(xtick_positions, unique_models, fontsize=14)
    
    plt.xlabel("Model", fontsize=16)
    plt.ylabel("Average FPS", fontsize=16)
    plt.legend(title='Central Unit', title_fontsize='13', fontsize='11')
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Pre-recorded Performance.png")
    plt.show()
    
    # Print the aggregated data in a LaTeX table format
    aggregated_perf['FPS'] = aggregated_perf.apply(lambda row: f"{row['Avg FPS']:.2f} $\\pm$ {row['Std FPS']:.2f}", axis=1)
    table_df = aggregated_perf[['Model', 'Central Unit', 'FPS']]
    
    latex_table = "\\begin{table}[h]\n\\centering\\caption{Performance comparison between models using different central units.}\n\\begin{tabular}{lcc}\n\\toprule\nModel & Central Unit Index & FPS \\\\\n\\midrule"
    
    # Group by 'Model' so that each model spans two rows in the table.
    for model, group in table_df.groupby('Model'):
        rows = group.to_dict('records')
        if len(rows) == 2:
            latex_table += f"\n\\multirow{{2}}{{*}}{{{model}}} & {rows[0]['Central Unit']} & {rows[0]['FPS']} \\\\"
            latex_table += f"\n & {rows[1]['Central Unit']} & {rows[1]['FPS']} \\\\{'[1ex]' if model == 'ArUco' else ''}"
        else:
            # Fallback if the group doesn't have exactly 2 entries.
            for i, row in enumerate(rows):
                if i == 0:
                    latex_table += f"\n{model} & {row['Central Unit']} & {row['FPS']} \\\\"
                else:
                    latex_table += f"\n & {row['Central Unit']} & {row['FPS']} \\\\"
                    
    latex_table += "\n\\bottomrule\n\\end{tabular}\n\\label{tab:pre-recorded-performance}\n\\end{table}"
    
    print(latex_table)
