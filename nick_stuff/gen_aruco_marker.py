import cv2
import numpy as np
import torch


def create_marker(aruco_dict, marker_id=23, marker_size=200, opacity=0.75):
    full_image = np.ones((marker_size+100, marker_size+100), dtype=np.uint8) * 255
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_image, 1)
    full_image[50:50+marker_size, 50:50+marker_size] = marker_image
    full_image[full_image == 0] += int(255 * (1 - opacity))
    cv2.imwrite(f"./nick_stuff/marker_id{marker_id}_sz{marker_size}_op{int(opacity*100)}.png", full_image)
    return full_image


if __name__ == '__main__':
    # Hyperparameters
    marker_id = 23
    marker_size = 200
    opacity = 0.75

    # Create the marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = create_marker(aruco_dict, marker_id, marker_size, opacity)

