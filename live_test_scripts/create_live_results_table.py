import os
import numpy as np
import pandas as pd

TABLE_HEADERS_BACKGROUND = [
    "Model", 
    "Background", 
    "FPS", "FPS Std Dev", 
    # "DPS", "DPS Std Dev", 
    # "Accuracy", "Accuracy Std Dev", 
    "F1 Score", "F1 Score Std Dev"
]
TABLE_HEADERS_LIGHTING = [
    "Model", 
    "Lighting", 
    "FPS", "FPS Std Dev", 
    # "DPS", "DPS Std Dev", 
    # "Accuracy", "Accuracy Std Dev", 
    "F1 Score", "F1 Score Std Dev"
]

def create_table(table_type, source_file="live_test_results/LiveResults_FULL (UPDATED with 5 RUNS).csv", decimal_places=2):

    if table_type == "background":
        table_headers = TABLE_HEADERS_BACKGROUND
        column_name = "Background"
        column_vals = ["Clean", "Noisy"]
        save_file = "LiveResults_TABLE_BACKGROUND.csv"
    elif table_type == "lighting":
        table_headers = TABLE_HEADERS_LIGHTING
        column_name = "Lighting"
        column_vals = ["Bright", "Dark"]
        save_file = "LiveResults_TABLE_LIGHTING.csv"
    else:
        raise ValueError(f"Invalid table type: {table_type}")

    # read full results CSV file
    df = pd.read_csv(source_file)

    # create new dataframe with only the columns we want
    table_df = pd.DataFrame(columns=table_headers)

    for val in column_vals:
        value_df = df.loc[df[column_name] == val]

        for model in ["YOLO", "ArUco"]:
            model_df = value_df.loc[value_df['Model'] == model]

            if model == "YOLO":
                row_data = [
                    model, 
                    val, 
                    np.round(model_df["Avg FPS"].mean(), decimal_places), 
                    np.round(model_df["Avg FPS"].std(), decimal_places), 
                    # np.round(model_df["Avg DPS"].mean(), decimal_places), 
                    # np.round(model_df["Avg DPS"].std(), decimal_places), 
                    # np.round(model_df["Accuracy"].mean(), decimal_places), 
                    # np.round(model_df["Accuracy"].std(), decimal_places), 
                    np.round(model_df["F1 Score"].mean(), decimal_places), 
                    np.round(model_df["F1 Score"].std(), decimal_places)
                ]
                table_df.loc[len(table_df)] = row_data 

            else:
                for m_opacity in range(25, 125, 25):
                    opacity_df = model_df.loc[model_df["Marker Opacity"] == m_opacity]
                    row_data = [
                        model+f" {str(m_opacity)}%",
                        val, 
                        np.round(opacity_df["Avg FPS"].mean(), decimal_places), 
                        np.round(opacity_df["Avg FPS"].std(), decimal_places), 
                        # np.round(opacity_df["Avg DPS"].mean(), decimal_places), 
                        # np.round(opacity_df["Avg DPS"].std(), decimal_places), 
                        # np.round(opacity_df["Accuracy"].mean(), decimal_places), 
                        # np.round(opacity_df["Accuracy"].std(), decimal_places), 
                        np.round(opacity_df["F1 Score"].mean(), decimal_places), 
                        np.round(opacity_df["F1 Score"].std(), decimal_places)
                    ]
                    table_df.loc[len(table_df)] = row_data 

    # save new table to CSV file
    table_df.to_csv(save_file, index=False)
    print(table_df.head(10))


if __name__ == '__main__':
    create_table(table_type="background")
    create_table(table_type="lighting")
