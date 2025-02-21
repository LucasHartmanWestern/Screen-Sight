import os
import pandas as pd
import numpy as np


if __name__ == '__main__':

    # open the main csv file
    csv_path = "live_test_results/LiveResults_FULL.csv"
    df = pd.read_csv(csv_path)

    for model in ["YOLO", "ArUco"]:
        model_df = df.loc[df['Model'] == model]
        print("================================================")
        print(f"[Model: {model}]")
        print(f"Mean FPS: {model_df['Avg FPS'].mean():.2f}")
        print(f"Mean F1 Score: {model_df['F1 Score'].mean():.2f}")

