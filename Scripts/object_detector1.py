import cv2
import numpy as np


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        imageLab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l_b = np.array([0, 14, 0])
        u_b = np.array([119, 255, 255])
        mask = frame[:].copy()
        imageRange = cv2.inRange(imageLab, l_b, u_b)



        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(imageRange, cv2.MORPH_CLOSE, kernel)
        denoised = cv2.medianBlur(closing, 5)

        # Find contours
        contours, _ = cv2.findContours(imageRange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 35000:
                objects_contours.append(cnt)

        return objects_contours
