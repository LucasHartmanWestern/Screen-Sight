import os
import pandas as pd
import numpy as np

'''
EXAMPLE:
882 frames, 894 detections

tp = 894
fp = 894 - 882 = 12
fn = 882 - 894 = -12 -> 0

tp / (tp + 0.5 * (fp + fn)) = f1_score
894 / (894 + 0.5 * (12 + 0)) = 0.986
'''

if __name__ == '__main__':

    # open the main csv file
    csv_path = "VideoResultsCSV.csv"
    df = pd.read_csv(csv_path)

    # calculate accuracy
    accuracies = []
    for index, row in df.iterrows():
        tp = row['Detections']
        total = row['Frames']
        accuracy = 1.00 if tp > total else np.round(tp / total, 2)
        accuracies.append(accuracy)
    df['Accuracy'] = accuracies

    # calculate the f1 scores
    f1_scores = []
    for index, row in df.iterrows():
        tp = row['Detections']
        fp = row['Detections'] - row['Frames'] if row['Detections'] > row['Frames'] else 0
        fn = row['Frames'] - row['Detections'] if row['Frames'] > row['Detections'] else 0
        f1_score = np.round(tp / (tp + (0.5 * (fp + fn))), 2)
        f1_scores.append(f1_score)
    df['F1Score'] = f1_scores

    # save to a new csv file
    print(df.head())
    df.to_csv('VideoResultsCSV_WithMetrics.csv', index=False)

