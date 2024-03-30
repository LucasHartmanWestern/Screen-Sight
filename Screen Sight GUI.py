import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab

show = False

# Find the scrcpy window
scrcpy_window = None
while scrcpy_window is None:
    windows = gw.getWindowsWithTitle("SCRCPY")
    if windows:
        scrcpy_window = windows[0]

def lerp(start, end, alpha):
    return (1 - alpha) * start + alpha * end


def interpolate_boxes(start_box, end_box, num_frames):
    interpolated_boxes = []
    for frame in range(num_frames):
        alpha = frame / (num_frames - 1) if num_frames > 1 else 1
        interpolated_box = {
            'x': lerp(start_box['x'], end_box['x'], alpha),
            'y': lerp(start_box['y'], end_box['y'], alpha),
            'w': lerp(start_box['w'], end_box['w'], alpha),
            'h': lerp(start_box['h'], end_box['h'], alpha),
        }
        interpolated_boxes.append(interpolated_box)
    return interpolated_boxes


def draw_phone_screen(frame, interpolated_box):
    # Capture the scrcpy window
    global scrcpy_window, show

    if not show: return

    # Extract the width, height, and top-left corner coordinates of the box
    box_w = int(interpolated_box['w'])
    box_h = int(interpolated_box['h'])
    x = int(interpolated_box['x'])
    y = int(interpolated_box['y'])

    left, top, width, height = scrcpy_window.box
    bbox = (left, top, left + width, top + height)
    screenshot = ImageGrab.grab(bbox)
    scrcpy_frame = np.array(screenshot)
    scrcpy_frame = cv2.cvtColor(scrcpy_frame, cv2.COLOR_RGB2BGR)

    # Calculate the aspect ratio of the phone screen
    aspect_ratio = width / height

    # Calculate new dimensions for the phone screen that preserve the aspect ratio
    # and fit within the bounding box
    if box_w / box_h > aspect_ratio:
        # If the bounding box is wider than the aspect ratio, adjust the width
        new_w = int(box_h * aspect_ratio)
        new_h = box_h
    else:
        # If the bounding box is taller than the aspect ratio, adjust the height
        new_w = box_w
        new_h = int(box_w / aspect_ratio)

    # Resize the captured phone screen to the new dimensions
    resized_phone_screen = cv2.resize(scrcpy_frame, (new_w, new_h))

    # Calculate the position to center the resized phone screen within the bounding box
    center_x = x + (box_w - new_w) // 2
    center_y = y + (box_h - new_h) // 2

    # Overlay the resized phone screen onto the frame at the centered position
    frame[center_y:center_y + new_h, center_x:center_x + new_w] = resized_phone_screen

def toggle_show():
    global show
    show = not show
    button_text.set("Hide Phone" if show else "Show Phone")

def main():
    frame_count = 0  # Initialize frame counter to track when to update visually.
    last_visual_detection = None  # The last detection that was visually updated.
    current_visual_detection = None  # The current target for visual updating.
    update_interval = 3  # Number of frames over which to interpolate after a new detection.

    # Load the YOLO model for object detection.
    model = YOLO("yolov8l.pt")

    for result in model.track(source=1, show=False, verbose=False, stream=True, agnostic_nms=True, hide_labels=True):
        frame = result.orig_img  # Get the original image frame from the result.

        # Convert the class IDs and bounding boxes to NumPy arrays
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        boxes_xywh = result.boxes.xywh.cpu().numpy()

        new_detection = None
        for cls_id, box in zip(class_ids, boxes_xywh):
            if cls_id == 67:  # Check if the class ID is 67 (e.g., for a specific object)
                x, y, w, h = box
                # Adjust x, y to be the top-left corner of the box
                x -= w / 2
                y -= h / 2
                new_detection = {'w': w, 'h': h, 'x': x, 'y': y}
                break

        if new_detection:
            last_visual_detection = current_visual_detection if current_visual_detection else new_detection
            current_visual_detection = new_detection
            frame_count = 0
        else:
            # If no new detection, consider removing the visual detection after the update interval
            if frame_count >= update_interval:
                current_visual_detection = None  # This will stop the box from being drawn

        if current_visual_detection:
            if frame_count < update_interval and last_visual_detection:
                alpha = frame_count / update_interval
                interpolated_box = {
                    'x': lerp(last_visual_detection['x'], current_visual_detection['x'], alpha),
                    'y': lerp(last_visual_detection['y'], current_visual_detection['y'], alpha),
                    'w': lerp(last_visual_detection['w'], current_visual_detection['w'], alpha),
                    'h': lerp(last_visual_detection['h'], current_visual_detection['h'], alpha),
                }
                draw_phone_screen(frame, interpolated_box)
            else:
                draw_phone_screen(frame, current_visual_detection)
            frame_count += 1

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()