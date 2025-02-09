import cv2
import numpy as np
import torch


def main():
    # Initialize the ArUco model
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params =  cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Initialize video capture from default camera (usually 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Count the number of detections that occur
    detection_counter = 0

    # Use OpenCV tick counts to compute FPS faster than time.time() and initialize for average FPS calculation
    frame_counter = 0
    elapsed_time = 0.0
    init_tick = cv2.getTickCount()

    # Run for 30 seconds
    while elapsed_time < 30.0:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame")
            break

        # Calculate FPS
        frame_counter += 1
        current_tick = cv2.getTickCount()
        elapsed_time = (current_tick - init_tick) / cv2.getTickFrequency()
        avg_fps = frame_counter / elapsed_time if elapsed_time > 0 else 0
        avg_dps = detection_counter / elapsed_time if elapsed_time > 0 else 0

        # Get show detection results
        (detections, ids, rejections) = aruco_detector.detectMarkers(frame)
        if len(detections) > 0:
            for markerCorners in detections:
                # extract the marker corners
                corners = markerCorners.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # draw the bounding box of the ArUCo detection
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                # increment detection counter
                detection_counter += 1

        # Overlay phone detection status on the frame using average FPS
        note_color = (0, 255, 0) if len(detections) > 0 else (0, 0, 255)
        note_text = f"FPS: {avg_fps:.1f}, DPS: {avg_dps:.1f}, Num Detections: {len(detections)}"
        cv2.putText(frame, note_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, note_color, 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Camera Feed', frame)
        
        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print metrics
    elapsed_time = (cv2.getTickCount() - init_tick) / cv2.getTickFrequency()
    print(f"Avg FPS: {(frame_counter / elapsed_time):.2f}") # frames per second
    print(f"Avg DPS: {(detection_counter / elapsed_time):.2f}") # detections per second
    print(f"Total num detections: {detection_counter}")
    print(f"Total num frames: {frame_counter}")
    print(f"Total time: {elapsed_time:.2f} seconds")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
