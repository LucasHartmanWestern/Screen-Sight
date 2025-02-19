import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

FILE_NAME_LIGHT_CLEAN_25 = "op25_light_clean.mp4"
FILE_NAME_LIGHT_CLEAN_50 = "op50_light_clean.mp4"
FILE_NAME_LIGHT_CLEAN_75 = "op75_light_clean.mp4"
FILE_NAME_LIGHT_CLEAN_100 = "op100_light_clean.mp4"

FILE_NAME_DARK_CLEAN_25 = "op25_dark_clean.mp4"
FILE_NAME_DARK_CLEAN_50 = "op50_dark_clean.mp4"
FILE_NAME_DARK_CLEAN_75 = "op75_dark_clean.mp4"
FILE_NAME_DARK_CLEAN_100 = "op100_dark_clean.mp4"

FILE_NAME_LIGHT_NOISY_25 = "op25_light_noisy.mp4"
FILE_NAME_LIGHT_NOISY_50 = "op50_light_noisy.mp4"
FILE_NAME_LIGHT_NOISY_75 = "op75_light_noisy.mp4"
FILE_NAME_LIGHT_NOISY_100 = "op100_light_noisy.mp4"

FILE_NAME_DARK_NOISY_25 = "op25_dark_noisy.mp4"
FILE_NAME_DARK_NOISY_50 = "op50_dark_noisy.mp4"
FILE_NAME_DARK_NOISY_75 = "op75_dark_noisy.mp4"
FILE_NAME_DARK_NOISY_100 = "op100_dark_noisy.mp4"

VIDEO_FILES_1 = [FILE_NAME_LIGHT_CLEAN_25, FILE_NAME_LIGHT_CLEAN_50, FILE_NAME_LIGHT_CLEAN_75, FILE_NAME_LIGHT_CLEAN_100]
VIDEO_FILES_2 = [FILE_NAME_DARK_CLEAN_25, FILE_NAME_DARK_CLEAN_50, FILE_NAME_DARK_CLEAN_75, FILE_NAME_DARK_CLEAN_100]
VIDEO_FILES_3 = [FILE_NAME_LIGHT_NOISY_25, FILE_NAME_LIGHT_NOISY_50, FILE_NAME_LIGHT_NOISY_75, FILE_NAME_LIGHT_NOISY_100]
VIDEO_FILES_4 = [FILE_NAME_DARK_NOISY_25, FILE_NAME_DARK_NOISY_50, FILE_NAME_DARK_NOISY_75, FILE_NAME_DARK_NOISY_100]

CSV_HEADERS = [
    "Model", "Central Unit", "Marker Opacity", "Background", "Lighting",
    "TP", "FP", "FN", "Frames", "Time", "Avg FPS", "Avg DPS", "Accuracy", "F1 Score"
]
    

def process_with_YOLO(dataframe):
    model = YOLO("yolo11n.pt")
    model_name = "YOLO"

    for video_set_num in range(1, 5):
        if video_set_num == 1:
            tqdm_desc = "[Part 1/8] YOLO Processing  "
            video_files_set = VIDEO_FILES_1
            background_conditions = "Clean"
            lighting_conditions = "Bright"
        elif video_set_num == 2:
            tqdm_desc = "[Part 2/8] YOLO Processing  "
            video_files_set = VIDEO_FILES_2
            background_conditions = "Clean"
            lighting_conditions = "Dark"
        elif video_set_num == 3:
            tqdm_desc = "[Part 3/8] YOLO Processing  "
            video_files_set = VIDEO_FILES_3
            background_conditions = "Noisy"
            lighting_conditions = "Bright"
        elif video_set_num == 4:
            tqdm_desc = "[Part 4/8] YOLO Processing  "
            video_files_set = VIDEO_FILES_4
            background_conditions = "Noisy"
            lighting_conditions = "Dark"
        else:
            print(f"Error: Invalid video files set number: {video_set_num}")
            return

        # loop through video files in set
        for i in tqdm(range(len(video_files_set)), desc=tqdm_desc):
            marker_opacity = 25 + (i * 25)
            fname = video_files_set[i]

            # read video file
            cap = cv2.VideoCapture(os.path.join("videos", fname))
            if not cap.isOpened():
                print(f"Error: Could not open video file: {fname}")
                return
            
            # total frames in video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(f"Total frames in video: {total_frames}")

            frame_counter = 0
            true_positives = 0 # number of detections
            false_positives = 0 # number of detections that should not be there
            false_negatives = 0 # number of times a detection was missed
            
            # loop through video frames
            start_time = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame")
                    break

                frame_counter += 1
                
                # get YOLO detection results
                result = model.predict(frame, conf=0.5, classes=[67], verbose=False)[0].boxes.xywh.cpu().numpy()
                if len(result) > 1:
                    false_positives += len(result) - 1 # unanticipated detections made
                    true_positives += 1 # correct detection made
                elif len(result) == 1:
                    true_positives += 1 # correct detection made
                else:
                    false_negatives += 1 # missed detection

                # end condition
                if frame_counter >= total_frames:
                    break

            # release resources
            cap.release()
            elapsed_time = time.time() - start_time

            # calculate metrics
            fps = frame_counter / elapsed_time
            dps = true_positives / elapsed_time
            acc = true_positives / frame_counter
            f1 = true_positives / (true_positives + (0.5 * (false_positives + false_negatives)))

            results_row = [
                model_name,
                name.lower(),
                marker_opacity,
                background_conditions,
                lighting_conditions,
                true_positives, # Total num true detections
                false_positives, # Total num false detections
                false_negatives, # Total num missed detections
                frame_counter, # Total num frames
                np.round(elapsed_time, 2), # Total time
                np.round(fps, 2), # Avg FPS
                np.round(dps, 2), # Avg DPS
                np.round(acc, 2), # Accuracy
                np.round(f1, 2) # F1 Score
            ]

            # append results to CSV dataframe
            dataframe.loc[len(dataframe)] = results_row

    return dataframe


def process_with_aruco(dataframe):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params =  cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    model_name = "ArUco"

    for video_set_num in range(1, 5):
        if video_set_num == 1:
            tqdm_desc = "[Part 5/8] ArUco Processing "
            video_files_set = VIDEO_FILES_1
            background_conditions = "Clean"
            lighting_conditions = "Bright"
        elif video_set_num == 2:
            tqdm_desc = "[Part 6/8] ArUco Processing "
            video_files_set = VIDEO_FILES_2
            background_conditions = "Clean"
            lighting_conditions = "Dark"
        elif video_set_num == 3:
            tqdm_desc = "[Part 7/8] ArUco Processing "
            video_files_set = VIDEO_FILES_3
            background_conditions = "Noisy"
            lighting_conditions = "Bright"
        elif video_set_num == 4:
            tqdm_desc = "[Part 8/8] ArUco Processing "
            video_files_set = VIDEO_FILES_4
            background_conditions = "Noisy"
            lighting_conditions = "Dark"
        else:
            print(f"Error: Invalid video files set number: {video_set_num}")
            return

        # loop through video files in set
        for i in tqdm(range(len(video_files_set)), desc=tqdm_desc):
            marker_opacity = 25 + (i * 25)
            fname = video_files_set[i]

            # read video file
            cap = cv2.VideoCapture(os.path.join("videos", fname))
            if not cap.isOpened():
                print(f"Error: Could not open video file: {fname}")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counter = 0
            true_positives = 0 # number of detections
            false_positives = 0 # number of detections that should not be there
            false_negatives = 0 # number of times a detection was missed
            
            # loop through video frames
            start_time = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame")
                    break

                frame_counter += 1
                
                # get aruco detection results
                _, ids, _ = aruco_detector.detectMarkers(frame)
                if ids is not None:
                    if len(ids) > 1:
                        false_positives += len(ids) - 1 # unanticipated detections made
                    true_positives += 1 # correct detection made
                else:
                    false_negatives += 1 # missed detection

                if frame_counter >= total_frames:
                    break

            # release resources 
            cap.release()
            elapsed_time = time.time() - start_time

            # calculate metrics
            fps = frame_counter / elapsed_time
            dps = true_positives / elapsed_time
            acc = true_positives / frame_counter
            f1 = true_positives / (true_positives + (0.5 * (false_positives + false_negatives)))
            
            results_row = [
                model_name,
                name.lower(),
                marker_opacity,
                background_conditions,
                lighting_conditions,
                true_positives, # Total num true detections
                false_positives, # Total num false detections
                false_negatives, # Total num missed detections
                frame_counter, # Total num frames
                np.round(elapsed_time, 2), # Total time
                np.round(fps, 2), # Avg FPS
                np.round(dps, 2), # Avg DPS
                np.round(acc, 2), # Accuracy
                np.round(f1, 2) # F1 Score
            ]

            # append results to CSV dataframe
            dataframe.loc[len(dataframe)] = results_row

    return dataframe


if __name__ == '__main__':
    # Must enter your name
    name = ""

    # process videos
    assert name.lower() in ["ethan", "lucas", "nick",], "Must enter your name on line 251!"
    df = pd.DataFrame(columns=CSV_HEADERS)
    df = process_with_YOLO(df)
    df = process_with_aruco(df)
    df.to_csv(f"video_test_results/VideoResults_{name.lower()}.csv", index=False)
