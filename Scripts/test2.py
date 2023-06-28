import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)

while True:
    frame = cv2.imread("../Images/slab2.jpg")

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)

    lower_thresh = cv2.getTrackbarPos("LH", "Tracking")
    upper_thresh = cv2.getTrackbarPos("LS", "Tracking")
    ret, thresh = cv2.threshold(blurred, lower_thresh, upper_thresh, cv2.THRESH_BINARY)
    cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    cv2.imshow("thresh", thresh)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()