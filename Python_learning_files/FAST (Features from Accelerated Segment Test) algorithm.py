import cv2
import numpy as np

image = cv2.imread('C:\\Users\\DELL\\Documents\\SEM 7\\EE604A(image Proccess)\\Assignment 1\\Assignment 1\\Q1\\test\\4.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# FAST threshold for corner detection (typically in the range of 10 to 30)
threshold = 10

# Non-maximal suppression (set to True to suppress non-maximum keypoints)
non_max_suppression = True
fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=non_max_suppression)
keypoints = fast.detect(gray, None)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
cv2.imshow('FAST Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
