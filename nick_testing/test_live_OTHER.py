import cv2
import time
from ultralytics import YOLO


def detect_phone(frame, model):
    result = model(frame)[0]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    boxes_xywh = result.boxes.xywh.cpu().numpy()
    for cls_id, box in zip(class_ids, boxes_xywh):
        # Check if the class ID corresponds to the desired object (e.g., ID 67 for phone)
        if cls_id == 67:
            x, y, w, h = box
            return [(int(x-(w/2)), int(y-(h/2))), (int(x+(w/2)), int(y+(h/2)))]  # [point1, point2]
    return None


def main():
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    # Initialize model
    model = YOLO("yolov8n.pt")

    # --- Main loop --- #
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (1750, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Detect phone
        phone_detection = detect_phone(frame, model)
        if phone_detection:
            cv2.rectangle(frame, phone_detection[0], phone_detection[1], (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Webcam Feed', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
