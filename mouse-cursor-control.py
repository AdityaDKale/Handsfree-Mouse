from imutils import face_utils
from utils import *
import numpy as np
import pyautogui as pyag
import imutils
import cv2
import os
import dlib
import collections
collections.Sequence = collections.abc.Sequence

# Thresholds and consecutive frame length for triggering the mouse action.
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSECUTIVE_FRAMES = 5
EYE_AR_THRESH = 0.24
EYE_AR_CONSECUTIVE_FRAMES = 7  # Reduced for faster response
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 3  # Reduced for faster response
WINK_MIN_DURATION = 3  # Minimum frames the eye should be closed for a valid wink

# Initialize the frame counters for each action as well as
# booleans used to indicate if action is performed or not
MOUTH_COUNTER = 0
EYE_COUNTER = 0
LEFT_WINK_COUNTER = 0
RIGHT_WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

# Load Haar Cascade face detector and Dlib's facial landmark predictor
haar_cascade_path = "haarcascade_frontalface_default.xml"

# Debug: Print the current working directory
print("Current Working Directory:", os.getcwd())

# Debug: Check if the Haar Cascade file exists
if not os.path.exists(haar_cascade_path):
    print(f"Error: Haar Cascade file not found at {haar_cascade_path}")
else:
    print(f"Haar Cascade file found at {haar_cascade_path}")

face_cascade = cv2.CascadeClassifier(haar_cascade_path)
shape_predictor_path = "model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(shape_predictor_path)

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Video capture
vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h

while True:
    # Grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # Determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    mouth = shape[mStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]

    # Because I flipped the frame, left is right, right is left.
    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    # Average the mouth aspect ratio together for both eyes
    mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    nose_point = (nose[3, 0], nose[3, 1])

    # Compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)
        
    # Check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if diff_ear > WINK_AR_DIFF_THRESH:
        if leftEAR < rightEAR:
            if leftEAR < EYE_AR_THRESH:
                LEFT_WINK_COUNTER += 1
                RIGHT_WINK_COUNTER = 0

                if LEFT_WINK_COUNTER > WINK_CONSECUTIVE_FRAMES and LEFT_WINK_COUNTER > WINK_MIN_DURATION:
                    pyag.click(button='left')
                    LEFT_WINK_COUNTER = 0
        elif leftEAR > rightEAR:
            if rightEAR < EYE_AR_THRESH:
                RIGHT_WINK_COUNTER += 1
                LEFT_WINK_COUNTER = 0

                if RIGHT_WINK_COUNTER > WINK_CONSECUTIVE_FRAMES and RIGHT_WINK_COUNTER > WINK_MIN_DURATION:
                    pyag.click(button='right')
                    RIGHT_WINK_COUNTER = 0
        else:
            LEFT_WINK_COUNTER = 0
            RIGHT_WINK_COUNTER = 0
    else:
        if ear <= EYE_AR_THRESH:
            EYE_COUNTER += 1

            if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                SCROLL_MODE = not SCROLL_MODE
                EYE_COUNTER = 0
        else:
            EYE_COUNTER = 0
            LEFT_WINK_COUNTER = 0
            RIGHT_WINK_COUNTER = 0

    if mar > MOUTH_AR_THRESH:
        MOUTH_COUNTER += 1

        if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
            INPUT_MODE = not INPUT_MODE
            MOUTH_COUNTER = 0
            ANCHOR_POINT = nose_point
    else:
        MOUTH_COUNTER = 0

    if INPUT_MODE:
        cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 100, 50
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

        dir = direction(nose_point, ANCHOR_POINT, w, h)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        drag = 18
        if dir == 'right':
            pyag.moveRel(drag, 0)
        elif dir == 'left':
            pyag.moveRel(-drag, 0)
        elif dir == 'up':
            if SCROLL_MODE:
                pyag.scroll(40)
            else:
                pyag.moveRel(0, -drag)
        elif dir == 'down':
            if SCROLL_MODE:
                pyag.scroll(-40)
            else:
                pyag.moveRel(0, drag)

    if SCROLL_MODE:
        cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

    cv2.putText(frame, "MAR: {:.2f}".format(mar), (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
    cv2.putText(frame, "Right EAR: {:.2f}".format(rightEAR), (460, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
    cv2.putText(frame, "Left EAR: {:.2f}".format(leftEAR), (460, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
    cv2.putText(frame, "Diff EAR: {:.2f}".format(np.abs(leftEAR - rightEAR)), (460, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `Esc` key was pressed, break from the loop
    if key == 27:
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vid.release()
