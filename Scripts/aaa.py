import numpy as np
import cv2
import math

img = cv2.imread('../Images/20230626_172306.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
l_b = np.array([0, 11, 74])
u_b = np.array([68, 255, 255])
mask = cv2.inRange(hsv, l_b, u_b)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the contour has four vertices (a rectangle)
    if len(approx) == 4:
        # Draw the contours on the original image
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

        # Get the four corners of the rect
        corners = approx.reshape(-1, 2)
print(corners)
print(corners[0])