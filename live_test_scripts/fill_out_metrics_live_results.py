import os
import pandas as pd
import numpy as np

'''
F1 Score Calculation:

F1 = D / (D + |F - D| / 2 )

Where:
D is Detections,
F is Frames

'''

if __name__ == '__main__':

    # open the main csv file
    csv_path = "live_test_results/LiveResults.csv"
    df = pd.read_csv(csv_path)

    # calculate the metrics
    df['Avg FPS'] = df['Detections'] / df['Time']
    df['Avg DPS'] = df['Detections'] / df['Time']
    df['Accuracy'] = df['Detections'] / df['Frames']
    df['F1Score'] = df['Detections'] / (df['Detections'] + (0.5 * np.abs(df['Frames'] - df['Detections'])))

    # round to 2 decimal places
    for col in ['Avg FPS', 'Avg DPS', 'Accuracy', 'F1 Score']:
        df[col] = df[col].apply(lambda x: np.round(x, 2))

    # save to a new csv file
    print(df.head())
    df.to_csv('live_test_results/LiveResults_FULL.csv', index=False)

