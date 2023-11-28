# python3 nhap1.py
import cv2
import numpy as np

def calculate_new_dimensions(w, h, angle):
    # Calculate the new dimensions of the image after rotation
    cos = np.abs(np.cos(np.radians(angle)))
    sin = np.abs(np.sin(np.radians(angle)))
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    return nW, nH

def rotate_image(image, angle, nW, nH):
    # Grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Grab the rotation matrix, then grab the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    #print(M)
    # Adjust the rotation matrixa to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# Load the images
img1 = cv2.imread('../Images/P1.jpg') #BinaryImage
img2 = cv2.imread('../Images/Pocket1-Trans.jpg') #OriginalImage
img3 = cv2.imread('../Images/W1.jpg') # BinarySlab

# Convert img1 to HSV and create a mask
hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
l_b = np.array([0, 0, 138])
u_b = np.array([255, 255, 255])
mask = cv2.inRange(hsv, l_b, u_b)

# Find contours in img1
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

# Get the angle of the first rectangle in img1
rect1 = cv2.minAreaRect(contours[0])
angle = rect1[-1]

# Calculate new dimensions for img1 and img2
nW1, nH1 = calculate_new_dimensions(img1.shape[1], img1.shape[0], angle)
nW2, nH2 = calculate_new_dimensions(img2.shape[1], img2.shape[0], angle)
nW3, nH3 = calculate_new_dimensions(img3.shape[1], img3.shape[0], angle)

# Rotate img1
rotated_img1 = rotate_image(img1, angle, nW1, nH1)

# Rotate img2
rotated_img2 = rotate_image(img2, angle, nW2, nH2)
# Rotate img3
rotated_img3 = rotate_image(img3, angle, nW3, nH3)

# Save the rotated image
cv2.imwrite('../Images/P22.jpg', rotated_img1)
cv2.imwrite('../Images/Pocket22-Trans.jpg', rotated_img2)
cv2.imwrite('../Images/W11.jpg', rotated_img3)

# Display the rotated images
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', img2)
cv2.namedWindow('Rotated Image 1', cv2.WINDOW_NORMAL)
cv2.imshow('Rotated Image 1', rotated_img1)
cv2.namedWindow('Rotated Image 2', cv2.WINDOW_NORMAL)
cv2.imshow('Rotated Image 2', rotated_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()