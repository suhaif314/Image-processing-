import cv2
import numpy as np
from PIL import Image, ImageDraw
from math import cos, sin, radians  # Import cos, sin, and radians
from matplotlib import pyplot as plt

#usage
def solution(image_path):
    image1= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################


    # Load and resize the image
    output_size = (600, 600)
    image = cv2.resize(image1, output_size)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area, assuming you want the largest one
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Approximate the largest contour to find corners
    largest_contour = contours[0]
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    #print(corners)
    # Prepare source and destination points for perspective transformation
    src_points = np.array([point[0] for point in corners], dtype=np.float32)

    dst_points = np.array([(0, 0), (output_size[0] , 0), (output_size[0] , output_size[1] ), (0, output_size[1] )], dtype=np.float32)

    #print(src_points)
    #print(dst_points)
    # Calculate the center point of src_points
    src_center = np.mean(src_points, axis=0)

    # Calculate the center point of dst_points
    dst_center = np.mean(dst_points, axis=0)

    # Calculate the angles of src_points with respect to the center
    src_angles = np.arctan2(src_points[:, 1] - src_center[1], src_points[:, 0] - src_center[0])

    # Calculate the angles of dst_points with respect to the center
    dst_angles = np.arctan2(dst_points[:, 1] - dst_center[1], dst_points[:, 0] - dst_center[0])

    # Sort src_points and dst_points based on angles
    src_points_sorted = src_points[np.argsort(src_angles)]
    dst_points_sorted = dst_points[np.argsort(dst_angles)]
    #print(src_points_sorted)
    #print(dst_points_sorted)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points_sorted, dst_points_sorted)

    # Apply the perspective transformation
    corrected_image = cv2.warpPerspective(image, matrix, output_size, borderMode=cv2.BORDER_REPLICATE)

    # Display the original and corrected images
    #plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Colored Flag (Original)")
    #plt.subplot(122), plt.imshow(cv2.cvtColor(corrected_image , cv2.COLOR_BGR2RGB)), plt.title("Corrected Flag")
    #plt.show()

    return corrected_image


