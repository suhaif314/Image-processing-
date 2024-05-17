import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    

    # Convert the input image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper HSV range for lava color (you may need to adjust these values)
    lower_range = np.array([0, 125, 125])
    upper_range = np.array([255, 255, 255])

    # Create a mask to segment the lava based on color
    color_mask = cv2.inRange(hsv_image, lower_range, upper_range)

    # Convert the input image to grayscale for easier processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment lava region
    ret, thresholded_image = cv2.threshold(gray_image, 90, 1000, cv2.THRESH_BINARY)
    
    # Perform morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    morphed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

    # return morphed_image
    # Combine the color-based mask and the morphology-based mask
    final_mask = cv2.bitwise_and(color_mask, morphed_image)
    # return final_mask
    # Find contours in the final mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask image with white as the detected lava region
    lava_result = np.zeros_like(image)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust the area threshold as needed
            cv2.drawContours(lava_result, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    image = cv2.cvtColor(lava_result, cv2.COLOR_BGR2RGB)






    ######################################################################  
    return image
