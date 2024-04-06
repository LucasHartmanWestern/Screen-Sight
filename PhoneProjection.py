from PIL import ImageGrab
import numpy as np
import cv2

def draw_phone_screen(frame, interpolated_box, scrcpy_window, show, current_orientation):
    if not show:
        return

    box_w = int(interpolated_box['w'])
    box_h = int(interpolated_box['h'])
    x = int(interpolated_box['x'])
    y = int(interpolated_box['y'])

    left, top, width, height = scrcpy_window.box
    bbox = (left, top, left + width, top + height)
    screenshot = ImageGrab.grab(bbox)
    scrcpy_frame = np.array(screenshot)
    scrcpy_frame = cv2.cvtColor(scrcpy_frame, cv2.COLOR_RGB2BGR)

    # Resize the phone screen to fit within the constant bounding box
    max_box_size = max(box_w, box_h)
    aspect_ratio = width / height
    if max_box_size / max_box_size > aspect_ratio:
        new_w = int(max_box_size * aspect_ratio)
        new_h = max_box_size
    else:
        new_w = max_box_size
        new_h = int(max_box_size / aspect_ratio)
    resized_phone_screen = cv2.resize(scrcpy_frame, (new_w, new_h))

    # Update the position to ensure the resized overlay stays centered
    center_x = x + (box_w - new_w) // 2
    center_y = y + (box_h - new_h) // 2

    # Rotate the frame based on the orientation
    rotation_angle = 90 - np.degrees(np.arctan2(current_orientation['y'], current_orientation['x']))
    rotation_matrix = cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[0] // 2), -rotation_angle, 1)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Rotate the center point of the bounding box
    center_point = np.array([center_x + new_w / 2, center_y + new_h / 2, 1])
    rotated_center = rotation_matrix.dot(center_point)

    # Adjust the position based on the rotated center
    x1 = max(int(rotated_center[0] - new_w / 2), 0)
    y1 = max(int(rotated_center[1] - new_h / 2), 0)
    x2 = min(x1 + new_w, frame.shape[1])
    y2 = min(y1 + new_h, frame.shape[0])

    # Place the rotated overlay onto the rotated frame
    rotated_frame[y1:y2, x1:x2] = resized_phone_screen[0:y2-y1, 0:x2-x1]

    # Rotate the frame back to the original orientation
    rotation_matrix = cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[0] // 2), rotation_angle, 1)
    rotated_frame = cv2.warpAffine(rotated_frame, rotation_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Define the padding size
    padding = 15

    top_bound = max(y - padding, 0)
    bottom_bound = min(y + box_h + padding, frame.shape[0])
    left_bound = max(x - padding, 0)
    right_bound = min(x + box_w + padding, frame.shape[1])

    # Copy the region inside the box with padding from rotated_frame to frame
    frame[top_bound:bottom_bound, left_bound:right_bound] = rotated_frame[top_bound:bottom_bound, left_bound:right_bound]