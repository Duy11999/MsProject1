import cv2
import numpy as np


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_b = np.array([0, 12, 51])
        u_b = np.array([15, 255, 255])

        # Create a Mask with adaptive threshold
        mask = cv2.inRange(hsv, l_b, u_b)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        denoised = cv2.medianBlur(closing, 5)

        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 35000:
                objects_contours.append(cnt)

        return objects_contours

