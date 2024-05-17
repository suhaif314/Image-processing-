import cv2
import numpy as np

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # Read the image file and convert it to RGB format
    i_img = cv2.imread(audio_path)
    i_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2RGB)

    # Convert the RGB to HSV color space
    hsv_image = cv2.cvtColor(i_img, cv2.COLOR_RGB2HSV)
    hue_base = hsv_image[:, :, 0]

    # Define a  thresholds for detecting Ravan's skin and necklace color this was done manually..
    ravan_skin_threshold = [6.25, 12.5] if np.median(hue_base) < 13 else [13, 16.5]
    ravan_necklace_threshold = [21, 24] if np.median(hue_base) < 13 else [17.5, 19]

    # Now we create  a mask for the necklace region based on the hue values we got 
    mask_img = np.where((hue_base >= ravan_necklace_threshold[0]) & (hue_base <= ravan_necklace_threshold[1]), 255, 0)

    # Excluding the black pixels from the mask based on the original image
    xb, yb = np.where(np.all(i_img == [0, 0, 0], axis=-1))
    x_coord_neck = np.mean(xb)
    mask_img[0:int(x_coord_neck), :] = 0

    # Display the necklace mask
    # plt.imshow(mask_img, cmap='gray')
    # plt.show()

    # Finding the coordinates of the white pixels in the mask
    x, y = np.where(mask_img == 255)
    y_centre = np.mean(y)
    #print(x_coord_neck, y_centre)

    # Create a mask for Ravan's skin based on the defined thresholds
    mask_img = np.where((hue_base >= ravan_skin_threshold[0]) & (hue_base <= ravan_skin_threshold[1]), 255, 0)
    mask_img[int(x_coord_neck):, :] = 0

    # Extracting the hue values from the top region of the image
    hue_top = hue_base[0:int(x_coord_neck), :]
    x_c, y_c = np.where((hue_top >= ravan_necklace_threshold[0]) & (hue_top <= ravan_necklace_threshold[1]))

    # Now Set a threshold for the top region of the necklace mask
    x_c = np.percentile(x_c, 80)
    mask_img[0:int(x_c), :] = 0
    # Applying  a Gaussian filter to the mean of the mask to identify heads
    sigma = 8.4
    gaussian = np.exp(-(np.linspace(-6, 6, 60) ** 2) / (sigma * i_img.shape[1] / 430) ** 2 / 2)
    gaussian = gaussian / sigma
    mean_face_mask = cv2.GaussianBlur(np.mean(mask_img, axis=0), (0, 0), sigma)
    # print(mean_face_mask)
    # Count the number of heads on the left and right sides of the image
    heads_left = sum(mean_face_mask[i] < mean_face_mask[i - 1] and mean_face_mask[i] < mean_face_mask[i + 1] for i in range(1, int(y_centre)))
    heads_right = sum(mean_face_mask[i] < mean_face_mask[i - 1] and mean_face_mask[i] < mean_face_mask[i + 1] for i in range(int(y_centre), len(mean_face_mask) - 1))

    print(heads_left, heads_right)
    if heads_left == 4 and heads_right == 5:
        class_name = 'real'
    else:
        class_name = 'fake'

    return class_name
