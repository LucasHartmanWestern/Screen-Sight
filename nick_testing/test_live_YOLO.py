import time
import cv2
from ultralytics import YOLO


def main():
    # Initialize variables
    model = YOLO("yolo11n.pt")  # best performance

    # Variables for FPS calculation
    prev_frame_time = 0.0
    new_frame_time = 0.0

    # Variables for avg fps calculation
    avg_fps = 0.0
    frame_count = 0

    # Variables for model speeds calculations
    preprocess_time = 0.0
    inference_time = 0.0
    postprocess_time = 0.0

    # Iterate over the tracking results from the YOLO model
    for result in model.track(source=0, show=False, verbose=False, stream=True, agnostic_nms=True):

        frame = result.orig_img
        frame_count += 1

        # Track model processing times
        preprocess_time += result.speed['preprocess']
        inference_time += result.speed['inference']
        postprocess_time += result.speed['postprocess']

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (1750, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        avg_fps += fps

        # Detection phone in frame (class id == 67)
        detection = None
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        for cls_id, box in zip(class_ids, boxes_xywh):
            if cls_id == 67:
                x, y, w, h = box
                detection = [(int(x-(w/2)), int(y-(h/2))), (int(x+(w/2)), int(y+(h/2)))]  # [point1, point2]
        
        # Draw phone detection button on the frame
        detection_note_text = "Phone detected" if detection else "Phone not detected"
        detection_note_color = (0, 255, 0) if detection else (0, 0, 255)
        cv2.putText(frame, detection_note_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, detection_note_color, 2, cv2.LINE_AA)

        # Draw the phone bbox on the frame if detected
        cv2.rectangle(frame, detection[0], detection[1], (0, 255, 0), 2) if detection else None

        # Display the frame in the window
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate averages
    print(f"Average FPS: {avg_fps/frame_count:.1f} frames per second")
    print(f"Average Preprocessing Time: {preprocess_time/frame_count:.1f} ms")
    print(f"Average Inference Time: {inference_time/frame_count:.1f} ms")
    print(f"Average Postprocessing Time: {postprocess_time/frame_count:.1f} ms")

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()