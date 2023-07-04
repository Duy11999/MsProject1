import cv2
import numpy as np

img = cv2.imread('../Images/slaba.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
l_b = np.array([0, 0, 212])
u_b = np.array([255, 255, 255])
mask = cv2.inRange(hsv, l_b, u_b)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if any contours were found
if len(contours) > 0:
    # Find the index of the largest contour
    largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    largest_contour = contours[largest_contour_index]

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Check if the contour has four vertices (a rectangle)
    if len(approx) == 4:
        # Draw the rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        corners = approx.reshape(-1, 2)
        print(corners)
        print(x)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
