import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    frame = cv2.imread("../Images/test12.jpg")

    imageLab  = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    mask = frame[:].copy()
    imageRange= cv2.inRange(imageLab, l_b, u_b)

    mask[:, :, 0] = imageRange
    mask[:, :, 1] = imageRange
    mask[:, :, 2] = imageRange


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    faceLab = cv2.bitwise_and(frame, mask)

    cv2.namedWindow(" imageRange ", cv2.WINDOW_NORMAL)
    cv2.imshow(" imageRange ", imageRange)

    cv2.namedWindow(" closing ", cv2.WINDOW_NORMAL)
    cv2.imshow(" closing ",  closing )

    cv2.namedWindow("faceLab", cv2.WINDOW_NORMAL)
    cv2.imshow("faceLab", faceLab)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()