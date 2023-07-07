import cv2

# load the image and display it
image = cv2.imread("../Images/slab2.jpg")
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# apply simple thresholding with a hardcoded threshold value T=230
(T, threshInv) = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
cv2.namedWindow("Simple Thresholding", cv2.WINDOW_NORMAL)
cv2.imshow("Simple Thresholding", threshInv)
cv2.waitKey(0)

# apply Otsu's automatic thresholding
(T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.namedWindow("Otsu Thresholding", cv2.WINDOW_NORMAL)
cv2.imshow("Otsu Thresholding", threshInv)

# instead of manually specifying the threshold value, we can use
# adaptive thresholding to examine neighborhoods of pixels and
# adaptively threshold each neighborhood
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,  51, 5)
cv2.namedWindow("Mean Adaptive Thresholding", cv2.WINDOW_NORMAL)
cv2.imshow("Mean Adaptive Thresholding", thresh)
cv2.waitKey(0)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
cv2.namedWindow("Gaussian Adaptive Thresholding", cv2.WINDOW_NORMAL)
cv2.imshow("Gaussian Adaptive Thresholding", thresh)
cv2.waitKey(0)