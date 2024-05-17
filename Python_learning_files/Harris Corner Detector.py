
import cv2
import numpy as np
image = cv2.imread('C:\\Users\\DELL\\Documents\\SEM 7\\EE604A(image Proccess)\\Assignment 1\\Assignment 1\\Q1\\test\\4.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


block_size = 2  # Size of the neighborhood for corner detection
ksize = 3  # Aperture parameter for the Sobel derivative
k = 0.04  # Harris detector free parameter


corners = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
print(corners)
# Threshold for an optimal value, it may vary depending on the image.
threshold = 0.01 * corners.max()

# Mark detected corners in green
image[corners > threshold] = [0, 255, 0]


cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
