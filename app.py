import cv2
from ultralytics import YOLO
import pygetwindow as gw
import PhoneSensors
from PhoneProjection import draw_phone_screen

# URL for connecting to the phone via WebSocket
phone_url = 'ws://<your_ip>:8080'

# Linear interpolation function
def lerp(start, end, alpha):
    # Returns the interpolated value between 'start' and 'end' using the parameter 'alpha'
    return (1 - alpha) * start + alpha * end

# Function to interpolate bounding boxes between two frames
def interpolate_boxes(start_box, end_box, num_frames):
    # List to store the interpolated boxes
    interpolated_boxes = []
    # Loop through each frame to calculate the interpolated box
    for frame in range(num_frames):
        # Calculate the interpolation parameter 'alpha'
        alpha = frame / (num_frames - 1) if num_frames > 1 else 1
        # Create the interpolated box for this frame
        interpolated_box = {
            'x': lerp(start_box['x'], end_box['x'], alpha),
            'y': lerp(start_box['y'], end_box['y'], alpha),
            'w': lerp(start_box['w'], end_box['w'], alpha),
            'h': lerp(start_box['h'], end_box['h'], alpha),
        }
        # Add the interpolated box to the list
        interpolated_boxes.append(interpolated_box)
    # Return the list of interpolated boxes
    return interpolated_boxes

# Function to toggle the visibility of the phone screen
def toggle_show():
    global show
    # Toggle the 'show' variable
    show = not show

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    # Check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click was within a specific rectangle (button area)
        if 10 <= x <= 110 and 10 <= y <= 50:
            # Toggle the visibility of the phone screen
            toggle_show()

# Initial state variables
detected = False
show = False

# Find the scrcpy window
scrcpy_window = None
while scrcpy_window is None:
    # Get the list of windows with the title "SCRCPY"
    windows = gw.getWindowsWithTitle("SCRCPY")
    # If there is at least one window, select the first one
    if windows:
        scrcpy_window = windows[0]

# Main function
def main():
    global detected, scrcpy_window, show

    # Initialize variables
    frame_count = 0
    last_visual_detection = None
    current_visual_detection = None
    update_interval = 3
    # Load the YOLO model
    model = YOLO("yolov8l.pt")

    # Iterate over the tracking results from the YOLO model
    for result in model.track(source=0, show=False, verbose=False, stream=True, agnostic_nms=True):
        # Get the original frame from the result
        frame = result.orig_img
        # Resize the frame
        frame = cv2.resize(frame, (1280, 960))

        # Get the class IDs and bounding boxes from the result
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        boxes_xywh = result.boxes.xywh.cpu().numpy()

        # Initialize the new detection variable
        new_detection = None
        # Iterate over the class IDs and boxes
        for cls_id, box in zip(class_ids, boxes_xywh):
            # Check if the class ID corresponds to the desired object (e.g., ID 67)
            if cls_id == 67:
                # Calculate the top-left corner coordinates and the size of the box
                x, y, w, h = box
                x -= w / 2
                y -= h / 2
                new_detection = {'w': w * 2, 'h': h * 2, 'x': x * 2, 'y': y * 2}
                # Break the loop after finding the first desired object
                break

        # Check if there is a new detection
        if new_detection:
            # Update the last and current visual detections
            last_visual_detection = current_visual_detection if current_visual_detection else new_detection
            current_visual_detection = new_detection
            # Reset the frame count
            frame_count = 0
        else:
            # If there is no new detection, update the current visual detection after a certain interval
            if frame_count >= update_interval:
                current_visual_detection = None

        # Check if there is a current visual detection
        if current_visual_detection:
            # Set the 'detected' flag to True
            detected = True
            # Get the current orientation of the phone
            current_orientation = PhoneSensors.orientation

            # Check if we need to interpolate between the last and current detections
            if frame_count < update_interval and last_visual_detection:
                # Calculate the interpolation parameter 'alpha'
                alpha = frame_count / update_interval
                # Interpolate the bounding box
                interpolated_box = {
                    'x': lerp(last_visual_detection['x'], current_visual_detection['x'], alpha),
                    'y': lerp(last_visual_detection['y'], current_visual_detection['y'], alpha),
                    'w': lerp(last_visual_detection['w'], current_visual_detection['w'], alpha),
                    'h': lerp(last_visual_detection['h'], current_visual_detection['h'], alpha),
                }
                # Draw the phone screen using the interpolated box
                draw_phone_screen(frame, interpolated_box, scrcpy_window, show, current_orientation)
            else:
                # If no interpolation is needed, draw the phone screen using the current detection
                draw_phone_screen(frame, current_visual_detection, scrcpy_window, show, current_orientation)
            # Increment the frame count
            frame_count += 1
        else:
            # If there is no current visual detection, set the 'detected' flag to False
            detected = False

        # Check if the phone was detected
        if detected:
            # Set the button text based on the visibility of the phone screen
            button_text = "Hide Phone" if show else "Show Phone"
            # Draw a rectangle (button) on the frame
            cv2.rectangle(frame, (10, 10), (110, 50), (255, 255, 255), -1)
            # Put the button text on the frame
            cv2.putText(frame, button_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # Display the frame in the window
        cv2.imshow('Frame', frame)
        cv2.setMouseCallback('Frame', mouse_callback)
        # Check for the 'q' key press to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    PhoneSensors.start_sensor_thread(phone_url)

    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()