import cv2
import numpy as np

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
lower_green = np.array([35, 100, 50])
upper_green = np.array([85, 255, 255])
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([140, 255, 255])

hsv_image = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
green_mask = cv2. inRange(hsv_image, lower_green, upper_green)
