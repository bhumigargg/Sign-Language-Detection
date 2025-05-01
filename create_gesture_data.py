import cv2
import numpy as np
import os
from datetime import datetime

# Base directory for all letters
base_output_dir = r"E:\\code\\output_images"

# Specify the current letter (update this for each gesture)
current_letter = 'a'  # Change this to 'b', 'c', etc., for different letters

# Create a subdirectory for the current letter
letter_dir = os.path.join(base_output_dir, current_letter)
os.makedirs(letter_dir, exist_ok=True)

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours for the image
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)


cam = cv2.VideoCapture(0)

num_frames = 0
num_imgs_taken = 0

while True:
    ret, frame = cam.read()

    # Flip the frame to prevent inverted image
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 60:
        cal_accum_avg(gray_frame, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    elif num_frames <= 300:
        hand = segment_hand(gray_frame)

        if hand is not None:
            thresholded, hand_segment = hand

            # Save the image in the corresponding letter directory
            if num_imgs_taken <= 300:
                cv2.imwrite(os.path.join(letter_dir, str(num_imgs_taken) + '.jpg'), thresholded)
                num_imgs_taken += 1
            else:
                break

    num_frames += 1

    # Display the frame
    cv2.imshow("Sign Detection", frame_copy)

    # Exit on pressing the Esc key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()