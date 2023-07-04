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
    frame = cv2.imread("../Images/20230629_142615.jpg")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=2)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.medianBlur(closing, 5)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.namedWindow("denoised", cv2.WINDOW_NORMAL)
    cv2.imshow("denoised", denoised)

    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.imshow("mask", mask)

    cv2.namedWindow("res", cv2.WINDOW_NORMAL)
    cv2.imshow("res", res)

    cv2.namedWindow("dilation", cv2.WINDOW_NORMAL)
    cv2.imshow("dilation", dilation)

    cv2.namedWindow("closing", cv2.WINDOW_NORMAL)
    cv2.imshow("closing", closing)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()