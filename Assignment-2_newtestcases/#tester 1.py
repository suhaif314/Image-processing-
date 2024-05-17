#tester 1

# import cv2

# # Load the pre-trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Read the image
# image = cv2.imread('C:\\Users\\DELL\\Documents\\SEM 7\\EE604A(image Proccess)\\Assignment-2\\Assignment-2\\Q3\\test\\r2.jpg')



# # Convert the image to grayscale (facial detection works better on grayscale images)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Perform face detection
# faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

# # Draw rectangles around the detected faces and print their coordinates
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     print(f"Face found at coordinates: x={x}, y={y}, width={w}, height={h}")

# # Display the result
# cv2.imshow('Detected Faces', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#tester 2

# import cv2
# import numpy as np

# # Read the image
# image = cv2.imread('C:\\Users\\DELL\\Documents\\SEM 7\\EE604A(image Proccess)\\Assignment-2\\Assignment-2\\Q3\\test\\r2.jpg')
# original_image = image.copy()

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply GaussianBlur to reduce noise and help the contour detection
# blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# # Use the Canny edge detector to find edges in the image
# edges = cv2.Canny(blurred_image, 50, 150)

# # Find contours in the edged image
# contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Filter out small contours that are unlikely to be faces
# min_contour_area = 150 

# filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# print(filtered_contours)
# # Draw rectangles around the detected faces and print their coordinates
# for i, contour in enumerate(filtered_contours):
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.putText(original_image, f"Face {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     print(f"Face {i+1} found at coordinates: x={x}, y={y}, width={w}, height={h}")

# # Display the result
# cv2.imshow('Detected Faces', original_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#tester 3 
import cv2
import dlib

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")

# Load the input image
image = cv2.imread("C:\\Users\\DELL\\Documents\\SEM 7\\EE604A(image Proccess)\\Assignment-2\\Assignment-2\\Q3\\test\\r2.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Loop through detected faces
for face in faces:
    # Get facial landmarks
    landmarks = predictor(gray, face)
    
    # Implement logic to count heads, detect real vs. fake Ravana, etc.

    # Example: Draw landmarks on the face
    for i in range(68):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Display the result
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
