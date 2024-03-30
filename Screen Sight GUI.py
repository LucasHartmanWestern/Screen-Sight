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
    global scrcpy_window, show
    if not show: return
    box_w = int(interpolated_box['w'])
    box_h = int(interpolated_box['h'])
    x = int(interpolated_box['x'])
    y = int(interpolated_box['y'])
    left, top, width, height = scrcpy_window.box
    bbox = (left, top, left + width, top + height)
    screenshot = ImageGrab.grab(bbox)
    scrcpy_frame = np.array(screenshot)
    scrcpy_frame = cv2.cvtColor(scrcpy_frame, cv2.COLOR_RGB2BGR)
    aspect_ratio = width / height
    if box_w / box_h > aspect_ratio:
        new_w = int(box_h * aspect_ratio)
        new_h = box_h
    else:
        new_w = box_w
        new_h = int(box_w / aspect_ratio)
    resized_phone_screen = cv2.resize(scrcpy_frame, (new_w, new_h))
    center_x = x + (box_w - new_w) // 2
    center_y = y + (box_h - new_h) // 2
    frame[center_y:center_y + new_h, center_x:center_x + new_w] = resized_phone_screen

def toggle_show():
    global show
    show = not show

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and 10 <= y <= 50:
            toggle_show()

def main():
    frame_count = 0
    last_visual_detection = None
    current_visual_detection = None
    update_interval = 3
    model = YOLO("yolov8l.pt")

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)

    for result in model.track(source=1, show=False, verbose=False, stream=True, agnostic_nms=True, hide_labels=True):
        frame = result.orig_img
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        new_detection = None
        for cls_id, box in zip(class_ids, boxes_xywh):
            if cls_id == 67:
                x, y, w, h = box
                x -= w / 2
                y -= h / 2
                new_detection = {'w': w, 'h': h, 'x': x, 'y': y}
                break

        if new_detection:
            last_visual_detection = current_visual_detection if current_visual_detection else new_detection
            current_visual_detection = new_detection
            frame_count = 0
        else:
            if frame_count >= update_interval:
                current_visual_detection = None

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

        button_text = "Hide Phone" if show else "Show Phone"
        cv2.rectangle(frame, (10, 10), (110, 50), (255, 255, 255), -1)
        cv2.putText(frame, button_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()