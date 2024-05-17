import cv2
import numpy as np
import matplotlib.pyplot as plt


def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    wt=0.2
    # Read images
    flash_img = cv2.imread(image_path_b).astype(np.float32)
    noflash_img = cv2.imread(image_path_a).astype(np.float32)

    # Decompose images into intensity and color
    flash_intensity = cv2.cvtColor(flash_img, cv2.COLOR_BGR2GRAY)
    noflash_intensity = cv2.cvtColor(noflash_img, cv2.COLOR_BGR2GRAY)

    flash_color = flash_img / (flash_intensity[:, :, np.newaxis] + 1e-8)
    noflash_color = noflash_img / (noflash_intensity[:, :, np.newaxis] + 1e-8)

    # Apply bilateral filter to get large-scale layers
    flash_large_scale = cv2.bilateralFilter(flash_intensity, d=0, sigmaColor=75, sigmaSpace=75)
    noflash_large_scale = cv2.bilateralFilter(noflash_intensity, d=0, sigmaColor=75, sigmaSpace=75)

    # Compute detail layers
    flash_detail = np.log10(flash_intensity + 1e-8) - np.log10(flash_large_scale + 1e-8)
    noflash_detail = np.log10(noflash_intensity + 1e-8) - np.log10(noflash_large_scale + 1e-8)

    # Recombine image in log domain
    result_detail = flash_detail
    result_large_scale = noflash_large_scale

    # Improve flash shadow
    result_large_scale = result_large_scale * (1 - flash_detail / np.max(flash_detail))
    result_detail = result_detail + flash_detail

    # White balance adjustment
    white_balance = np.mean(flash_color, axis=(0, 1)) / (np.mean(noflash_color, axis=(0, 1)) + 1e-8)
    result_color = flash_color * white_balance * wt + noflash_color * (1 - wt)

    # Recombine images in the log domain
    result_intensity = result_detail + np.log10(result_large_scale + 1e-8)
    result_img = result_color * (10 ** result_intensity[:, :, np.newaxis])

    # Clip values to valid image range
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)


    return result_img

