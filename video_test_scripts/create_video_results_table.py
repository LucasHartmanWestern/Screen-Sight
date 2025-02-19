import os
import numpy as np
import pandas as pd

CSV_LIST = [
    "video_test_results/VideoResults_nick.csv",
    "video_test_results/VideoResults_ethan.csv",
    "video_test_results/VideoResults_lucas.csv"
]

FULL_CSV_HEADERS = [
    "Model", "Central Unit", "Marker Opacity", "Background", "Lighting",
    "TP", "FP", "FN", "Frames", "Time", "Avg FPS", "Avg DPS", "Accuracy", "F1 Score"
]

TABLE_HEADERS = [
    "Model", "Central Unit", "FPS", "FPS Std Dev", "DPS", "DPS Std Dev", "Accuracy", "Accuracy Std Dev", "F1 Score", "F1 Score Std Dev"
]


def create_full_csv():
    # create main dataframe
    df = pd.DataFrame(columns=FULL_CSV_HEADERS)

    # loop through CSV files
    for i in range(len(CSV_LIST)):
        csv_file = CSV_LIST[i]

        # check if file exists
        if not os.path.exists(csv_file):
            print(f"Error: File {csv_file} does not exist")
            quit()

        # append to main dataframe
        df_temp = pd.read_csv(csv_file)
        df = pd.concat([df, df_temp], ignore_index=True)

    # save main dataframe to CSV file
    df.to_csv(f"video_test_results/VideoResults_ALL.csv", index=False)


def create_table(save_file="VideoResults_TABLE.csv", source_file="video_test_results/VideoResults_ALL.csv", decimal_places=2):
    # read full results CSV file
    df = pd.read_csv(source_file)

    # create new dataframe with only the columns we want
    table_df = pd.DataFrame(columns=TABLE_HEADERS)

    for model in ["YOLO", "ArUco"]:
        model_df = df.loc[df['Model'] == model]

        for central_unit in ["nick", "ethan", "lucas"]:
            cu_df = model_df.loc[model_df['Central Unit'] == central_unit]
            row_data = [
                model, 
                central_unit, 
                np.round(cu_df["Avg FPS"].mean(), decimal_places), 
                np.round(cu_df["Avg FPS"].std(), decimal_places), 
                np.round(cu_df["Avg DPS"].mean(), decimal_places), 
                np.round(cu_df["Avg DPS"].std(), decimal_places), 
                np.round(cu_df["Accuracy"].mean(), decimal_places), 
                np.round(cu_df["Accuracy"].std(), decimal_places), 
                np.round(cu_df["F1 Score"].mean(), decimal_places), 
                np.round(cu_df["F1 Score"].std(), decimal_places)
            ]
            table_df.loc[len(table_df)] = row_data 

    # save new table to CSV file
    table_df.to_csv(save_file, index=False)
    print(table_df.head(10))


if __name__ == '__main__':
    create_full_csv()
    create_table()
