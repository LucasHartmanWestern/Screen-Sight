import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO
import numpy as np
import torch

def main():
    # Initialize the YOLO model without specifying the device (it will automatically select GPU if available)
    model = YOLO("yolo11n.pt")
    
    # Check if GPU is available and print which device is being used
    if torch.cuda.is_available():
        print("GPU is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU not available. Using CPU.")
    
    # Additionally, print the device of the model parameters (if available)
    try:
        model_device = next(model.model.parameters()).device
        print("Model is running on device:", model_device)
    except Exception as e:
        print("Could not determine model device:", e)
    
    # Use OpenCV tick counts to compute FPS faster than time.time() and initialize for average FPS calculation
    init_tick = cv2.getTickCount()
    frame_counter = 0

    # Variables for model speeds calculations (if needed later)
    preprocess_time = 0.0
    inference_time = 0.0
    postprocess_time = 0.0

    # Iterate over the tracking results from the YOLO model
    for result in model.track(source=0, show=False, verbose=False, stream=True, agnostic_nms=True):
        frame = result.orig_img

        frame_counter += 1

        # Accumulate model processing times
        preprocess_time += result.speed.get('preprocess', 0)
        inference_time += result.speed.get('inference', 0)
        postprocess_time += result.speed.get('postprocess', 0)

        # Calculate average FPS using cv2.getTickCount
        current_tick = cv2.getTickCount()
        elapsed_time = (current_tick - init_tick) / cv2.getTickFrequency()
        avg_fps = frame_counter / elapsed_time if elapsed_time > 0 else 0

        # Detect phone in frame (class id == 67) using vectorized operations
        detection = None
        class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        mask = class_ids == 67
        if mask.any():
            box = boxes_xywh[mask][0]
            x, y, w, h = box
            detection = [(int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2))]

        # Overlay phone detection status on the frame using average FPS
        note_text = f"Phone detected: {avg_fps:.1f} fps" if detection else "Phone not detected"
        note_color = (0, 255, 0) if detection else (0, 0, 255)

        cv2.putText(frame, note_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, note_color, 2, cv2.LINE_AA)

        # Draw the phone bounding box if detection exists
        if detection:
            cv2.rectangle(frame, detection[0], detection[1], (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()