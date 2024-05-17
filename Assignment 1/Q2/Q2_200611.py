import matplotlib.pyplot as plt
import cv2
import numpy as np
import librosa

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    #print(y,sr)

    # Compute the spectrogram
    n_fft = 2048
    hop_length = 512
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)

    #print(spec)

    # Convert power spectrogram to decibel scale
    spec_db = librosa.power_to_db(spec, ref=np.max)
    #plt.imshow(spec_db)
    #print(spec_db)
    # Now Calculate the mean decibel value for the entire spectrogram
    mean_db = np.mean(spec_db)
    #print(mean_db)

    #---------------------------------------------
    # Apply thresholding
    threshold = -25  # Adjust this threshold as needed
    binary_image = np.where(spec_db >= threshold, 255, 0).astype(np.uint8)

    #plot the binary image

    #plt.imshow(binary_image, cmap='gray')

    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    
    #plot the dilated image
    #plt.imshow(dilated_image, cmap='gray')

    # Find connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_image, connectivity=8)
    #print(num_labels,labels,stats,centroids)
    # Remove the background component (largest region)

    if num_labels > 1:
        largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mel_region = labels == largest_component_idx
    else:
        mel_region = labels == 1  # Use the only component as metal

    #print(mel_region,np.sum(mel_region))

    # Define a criterion for classification (e.g., size of the detected region)
    min_mel_region_size = 4000  # Adjust this value as needed

    # Classify based on the size of the detected region
    if np.sum(mel_region) >= min_mel_region_size:
        class_name = 'metal'
    else:
        class_name = 'cardboard'

    return class_name

