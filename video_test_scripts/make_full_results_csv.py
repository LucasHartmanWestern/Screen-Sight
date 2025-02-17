import os
import numpy as np
import pandas as pd

CSV_LIST = [
    "video_test_results/VideoResults_nick.csv",
    "video_test_results/VideoResults_ethan.csv",
    "video_test_results/VideoResults_lucas.csv"
]

CSV_HEADERS = [
    "Model", "Central Unit", "Marker Opacity", "Background", "Lighting",
    "TP", "FP", "FN", "Frames", "Time", "Avg FPS", "Avg DPS", "Accuracy", "F1 Score"
]


if __name__ == '__main__':
    # create main dataframe
    df = pd.DataFrame(columns=CSV_HEADERS)

    # loop through CSV files
    for i in range(len(CSV_LIST)):
        csv_file = CSV_LIST[i]

        # check if file exists
        if not os.path.exists(csv_file):
            print(f"Error: File {csv_file} does not exist")
            quit()

        # read CSV file
        df_temp = pd.read_csv(csv_file)

        # change all values under "Central Unit" column to the name of the CSV file
        df_temp["Central Unit"] = int(i)

        # append to main dataframe
        df = pd.concat([df, df_temp], ignore_index=True)

    # save main dataframe to CSV file
    df.to_csv(f"video_test_results/VideoResults_ALL.csv", index=False)

