import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

def solution(image_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    image = cv2.imread(image_path)
    
    # Converting the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying edge detection to find lines in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Finding lines in the edge-detected image by Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    print(lines)
    # Calculate the angle of the most prominent line detected by having median of all lines detected.
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
        predominant_angle = np.median(angles)
    else:
        predominant_angle = 0
    print(angles)
    # Rotating the image to the required angle
    angle = predominant_angle * (180 / math.pi)  # Angle in degrees
    if angle < 0:
        angle += 180
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Creating a white background image with the same dimensions as the original image
    #white_background = np.ones_like(image) * 255

    # Rotate the original image and overlay it onto the white background
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=0.9)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Display the original and aligned images side by side
    #plt.figure(figsize=(14, 7))
    #plt.subplot(1, 2, 1)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.title("Original Image")

    #plt.subplot(1, 2, 2)
    #plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    #plt.title("Aligned Image")

    #plt.show()

    return rotated_image
