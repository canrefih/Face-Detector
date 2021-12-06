import cv2
from random import randrange

# This will capture the faces from webcame

# importing the face detecting library - you can check out from docs.opencv CascadeClassifier btw
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# to capture video from webcam(to ensure that we use (0)), we can use as ('sample.mp4') either
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:

    # reading the current frame (boolean, current frame)
    successful_frame_read, frame = webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    for(x, y, w, h) in face_coordinates:
        # x,y coordinates - width and height - RGB(random colours) - thickness of the square
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('Face Detector', frame)
    
    # when we gave it (1) it will capture the frame on every millisecond
    key = cv2.waitKey(1)

    # it will stop when we press the Q - 81 and 113 uppercase and lowercase Q/q ASCII codes
    if key==81 or key==113:
        break

# clearing the webcam data and to let it closed
webcam.release()

"""
# This will capture the faces from images

# importing the face detecting library
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choosing the image to being detected
img = cv2.imread('mutli.jpg')

# must be converted to the one type scale as one gray, red or blue scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detecting the faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# assigning the coordinates to the variables (capturing the all faces on image)
for(x, y, w, h) in face_coordinates:
    # x,y coordinates - width and height - RGB(random colours) - thickness of the square
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# display the image with the faces
cv2.imshow('Face Detector', img)

# ensuring the function to being closed with a key pressed (for example enter to exit)
cv2.waitKey()
"""

