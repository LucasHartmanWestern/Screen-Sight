import argparse
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
    "Trial", "Model", "Central Unit", "Marker Opacity", "Background", "Lighting",
    "TP", "FP", "FN", "Frames", "Time", "Avg FPS", "Avg DPS", "Accuracy", "F1 Score"
]


def check_video_paths():
    print("Checking video paths...")
    for trial in range(1, 4):
        for video_set in [VIDEO_FILES_1, VIDEO_FILES_2, VIDEO_FILES_3, VIDEO_FILES_4]:
            for fname in video_set:
                video_path = os.path.join("videos", f"trial_{trial}", fname)
                if not os.path.exists(video_path):
                    print(f"Error: Video file {video_path} does not exist, please ensure all videos are present in the videos folder")
                    return False
    print("All video paths checked successfully.")
    return True
    

def process_with_YOLO(dataframe, tester_name, trial_num):
    model = YOLO("yolo11n.pt")
    model_name = "YOLO"

    for video_set_num in range(1, 5):
        if video_set_num == 1:
            tqdm_desc = f"[Part {1+((trial_num-1)*8)}/24] YOLO Processing  "
            video_files_set = VIDEO_FILES_1
            background_conditions = "Clean"
            lighting_conditions = "Bright"
        elif video_set_num == 2:
            tqdm_desc = f"[Part {2+((trial_num-1)*8)}/24] YOLO Processing  "
            video_files_set = VIDEO_FILES_2
            background_conditions = "Clean"
            lighting_conditions = "Dark"
        elif video_set_num == 3:
            tqdm_desc = f"[Part {3+((trial_num-1)*8)}/24] YOLO Processing  "
            video_files_set = VIDEO_FILES_3
            background_conditions = "Noisy"
            lighting_conditions = "Bright"
        elif video_set_num == 4:
            tqdm_desc = f"[Part {4+((trial_num-1)*8)}/24] YOLO Processing  "
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
            video_path = os.path.join("videos", f"trial_{trial_num}", fname)

            # read video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
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
                trial_num,
                model_name,
                tester_name.lower(),
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


def process_with_aruco(dataframe, tester_name, trial_num):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params =  cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    model_name = "ArUco"

    for video_set_num in range(1, 5):
        if video_set_num == 1:
            tqdm_desc = f"[Part {5+((trial_num-1)*8)}/24] ArUco Processing "
            video_files_set = VIDEO_FILES_1
            background_conditions = "Clean"
            lighting_conditions = "Bright"
        elif video_set_num == 2:
            tqdm_desc = f"[Part {6+((trial_num-1)*8)}/24] ArUco Processing "
            video_files_set = VIDEO_FILES_2
            background_conditions = "Clean"
            lighting_conditions = "Dark"
        elif video_set_num == 3:
            tqdm_desc = f"[Part {7+((trial_num-1)*8)}/24] ArUco Processing "
            video_files_set = VIDEO_FILES_3
            background_conditions = "Noisy"
            lighting_conditions = "Bright"
        elif video_set_num == 4:
            tqdm_desc = f"[Part {8+((trial_num-1)*8)}/24] ArUco Processing "
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
            video_path = os.path.join("videos", f"trial_{trial_num}", fname)

            # read video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
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
                trial_num,
                model_name,
                tester_name.lower(),
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
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True, help="Name of the person running the script")
    args = parser.parse_args()
    name = args.n.lower()
    assert name in ["ethan", "lucas", "nick",], f"Incorrect name argument: {name}. Must be one of: ethan, lucas, nick."

    # check video paths
    assert check_video_paths(), "Video paths check failed, please ensure all videos are present in the videos folder"

    # process videos
    df = pd.DataFrame(columns=CSV_HEADERS)
    for trial in range(1, 4):
        df = process_with_YOLO(df, tester_name=name, trial_num=trial)
        df = process_with_aruco(df, tester_name=name, trial_num=trial)
    
    # save completed dataframe to CSV file
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/VideoResults_{name.lower()}.csv", index=False)
