import os
import cv2
import numpy as np
import torch
import time
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


def create_folders():
    os.makedirs("video_test_results", exist_ok=True)
    os.makedirs("video_test_results/yolo", exist_ok=True)
    os.makedirs("video_test_results/aruco", exist_ok=True)


def process_with_YOLO(video_files_set_num, verbose=False):
    model = YOLO("yolo11n.pt")

    if video_files_set_num == 1:
        video_files_set = VIDEO_FILES_1
        results_file_name = "yolo_video_results_1_LIGHT_CLEAN.txt"
    elif video_files_set_num == 2:
        video_files_set = VIDEO_FILES_2
        results_file_name = "yolo_video_results_2_DARK_CLEAN.txt"
    elif video_files_set_num == 3:
        video_files_set = VIDEO_FILES_3
        results_file_name = "yolo_video_results_3_LIGHT_NOISY.txt"
    elif video_files_set_num == 4:
        video_files_set = VIDEO_FILES_4
        results_file_name = "yolo_video_results_4_DARK_NOISY.txt"
    else:
        print(f"Error: Invalid video files set number: {video_files_set_num}")
        return
    
    # create results file
    open(os.path.join("video_test_results", "yolo", results_file_name), "w+").close()

    # loop through video files in set
    for fname in video_files_set:

        # read video file
        cap = cv2.VideoCapture(os.path.join("videos", fname))
        if not cap.isOpened():
            print(f"Error: Could not open video file: {fname}")
            return
        
        # total frames in video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f"Total frames in video: {total_frames}")

        frame_counter = 0
        detection_counter = 0
        
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
            if len(result) > 0:
                for box in result:
                    x, y, w, h = box
                    cv2.rectangle(frame, (int(x-(w/2)), int(y-(h/2))), (int(x+(w/2)), int(y+(h/2))), (0, 255, 0), 2)
                    detection_counter += 1

            if frame_counter >= total_frames:
                break

        # release resources
        cap.release()
        
        # print metrics
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Avg FPS: {(frame_counter / elapsed_time):.2f}")
            print(f"Avg DPS: {(detection_counter / elapsed_time):.2f}")
            print(f"Total num frames: {frame_counter}")
            print(f"Total num detections: {detection_counter}")
            print(f"Total time: {elapsed_time:.2f} seconds")

        # write results to .txt file
        with open(os.path.join("video_test_results", "yolo", results_file_name), "a") as f:
            f.write(f"\n----- File name: {fname} -----\n")
            f.write(f"Avg FPS: {(frame_counter / elapsed_time):.2f}\n")
            f.write(f"Avg DPS: {(detection_counter / elapsed_time):.2f}\n")
            f.write(f"Total num frames: {frame_counter}\n")
            f.write(f"Total num detections: {detection_counter}\n")
            f.write(f"Total time: {elapsed_time:.2f} seconds\n")


def process_with_aruco(video_files_set_num, verbose=False):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params =  cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    if video_files_set_num == 1:
        video_files_set = VIDEO_FILES_1
        results_file_name = "aruco_video_results_1_LIGHT_CLEAN.txt"
    elif video_files_set_num == 2:
        video_files_set = VIDEO_FILES_2
        results_file_name = "aruco_video_results_2_DARK_CLEAN.txt"
    elif video_files_set_num == 3:
        video_files_set = VIDEO_FILES_3
        results_file_name = "aruco_video_results_3_LIGHT_NOISY.txt"
    elif video_files_set_num == 4:
        video_files_set = VIDEO_FILES_4
        results_file_name = "aruco_video_results_4_DARK_NOISY.txt"
    else:
        print(f"Error: Invalid video files set number: {video_files_set_num}")
        return
    
     # create results file
    open(os.path.join("video_test_results", "aruco", results_file_name), "w+").close()

    # loop through video files in set
    for fname in video_files_set:

        # read video file
        cap = cv2.VideoCapture(os.path.join("videos", fname))
        if not cap.isOpened():
            print(f"Error: Could not open video file: {fname}")
            return
        
        # total frames in video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f"Total frames in video: {total_frames}")
        
        frame_counter = 0
        detection_counter = 0
        
        # loop through video frames
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break

            frame_counter += 1
            
            # get aruco detection results
            corners, ids, rejected = aruco_detector.detectMarkers(frame)
            if ids is not None:
                detection_counter += len(ids)

            if frame_counter >= total_frames:
                break

        # release resources 
        cap.release()
        
        # print metrics
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Avg FPS: {(frame_counter / elapsed_time):.2f}")
            print(f"Avg DPS: {(detection_counter / elapsed_time):.2f}")
            print(f"Total num frames: {frame_counter}")
            print(f"Total num detections: {detection_counter}")
            print(f"Total time: {elapsed_time:.2f} seconds")

        # write results to .txt file
        with open(os.path.join("video_test_results", "aruco", results_file_name), "a") as f:
            f.write(f"\n----- File name: {fname} -----\n")
            f.write(f"Avg FPS: {(frame_counter / elapsed_time):.2f}\n")
            f.write(f"Avg DPS: {(detection_counter / elapsed_time):.2f}\n")
            f.write(f"Total num frames: {frame_counter}\n")
            f.write(f"Total num detections: {detection_counter}\n")
            f.write(f"Total time: {elapsed_time:.2f} seconds\n")


if __name__ == '__main__':
    try:
        create_folders()
        for set_num in tqdm(range(1, 5), desc="Processing video sets"):
            process_with_YOLO(set_num)
            process_with_aruco(set_num)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
