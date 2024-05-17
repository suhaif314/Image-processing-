import cv2
import numpy as np
# Function to compute bilateral filter on an image
def compute_bilateral_filter(image, spatial_sig, range_sig):
    # Set up parameters for bilateral filter
    kernal_size = int(spatial_sig) * 4
    kernal_size = int(kernal_size) + 1
    rand_samp = int(kernal_size) # random pixel sampling and image padding.
    k_half = kernal_size // 2

    # Pad image for convolution
    conv_img_padd = np.pad(image, ((k_half - 1, k_half - 1), (k_half - 1, k_half - 1), (0, 0)), mode='constant')

    # Pad the input image with zeros to handle border pixels during convolution.
    # Randomly sample pixels
    samp_pix = np.random.randint(low=0, high=kernal_size ** 2, size=(rand_samp,), dtype=int)
    i = samp_pix // kernal_size
    j = samp_pix - i * kernal_size
    J = np.array([i - 1, j - 1])
    I = np.array([[k_half], [k_half]])

    #Randomly sample pixels within the filter kernel
    # Compute spatial Gaussian weights
    gauss_s = np.exp(-np.linalg.norm(np.transpose(I) - np.transpose(J)) * 2 / (2 * spatial_sig ** 2))
    gaussian_s = np.repeat(gauss_s, 3, axis=0)
    de_normztion = 2 * range_sig ** 2

    # Now Compute spatial Gaussian weights based on the Euclidean distance between sampled pixels.
    # Filtered image initialization
    filter_img_final = np.zeros_like(image)
    
    #Loop over image pixels, taking into account the padded borders.

    for n in range(k_half - 1, image.shape[0] + k_half - 1):
        n1 = n + i - 1
        n2 = n + k_half
        for m in range(k_half - 1, image.shape[1] + k_half - 1):
            m1 = m + j - 1

            # Check if indices are within bounds
            if n1.min() < 0 or n1.max() >= conv_img_padd.shape[0] or m1.min() < 0 or m1.max() >= conv_img_padd.shape[1]:
                continue

            # Compute range Gaussian weights
            gaussian = np.exp(-(conv_img_padd[n2, m] - conv_img_padd[n1, m1]) ** 2 / de_normztion)
            weights = np.multiply(gaussian_s, gaussian)

            # Apply bilateral filter
            pix = np.multiply(conv_img_padd[n1, m1], weights)
            filter_img_final[n - k_half + 1, m - k_half + 1, :] = np.divide(np.sum(pix, axis=0), np.sum(weights, axis=0))

    return filter_img_final

# Function to process flash and noflash images
def processing_images(flash_img, noflash_img, f_range, nf_range):
    G = np.ones_like(flash_img) * 0.02

    # Bilateral filtering for base images
    flash_base = compute_bilateral_filter(flash_img, 7, f_range * 0.1)
    non_flash_base = compute_bilateral_filter(noflash_img, 7, nf_range * 0.1)

    # Bilateral filtering for detail images
    # Compute detail images (F_detail) by dividing the flash image by the corresponding base image.
    flash_detial = np.divide(flash_img + G, flash_base + G)

    # Additional bilateral filtering for noflash_img and flash_img
    noflash_detial = compute_bilateral_filter(noflash_img, 7, f_range * 0.001)  # Fixed number of arguments here

    # Binary mask M
    mask_b = np.where((np.linalg.norm(flash_img - noflash_img) < 90) & (np.linalg.norm(flash_img - noflash_img) > 300), 1, 0)
    mask_b = mask_b.astype(np.float32)

    # Morphological operations on the mask M
    kernel = np.ones((5, 5), np.uint8)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kernel)
    mask_b = cv2.blur(mask_b, (5, 5))

    # Compute final image
    comb_img = np.multiply(1 - mask_b, noflash_detial)
    img_final = np.multiply(comb_img, flash_detial) + np.multiply(mask_b, non_flash_base)

    return img_final


def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    image_no_flash = cv2.imread(image_path_a)
    image_flash = cv2.imread(image_path_b)
    range_NF = image_no_flash.max() - image_no_flash.min()
    range_F = image_flash.max() - image_flash.min()
    image_no_flash = image_no_flash[:, :, [2, 1, 0]]
    image_flash = image_flash[:, :, [2, 1, 0]]
    image = processing_images(image_flash, image_no_flash, range_F, range_NF)

    return image


