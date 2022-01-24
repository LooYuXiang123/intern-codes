#Loading the Libraries
import sys
import cv2
from IPython.display import Video
import os
import numpy as np
import logging

# py video2frames.py ./Video_Dataset/Test_data/test_data.mp4 ./Video_Dataset/Test_data/frames/

input_filename = sys.argv[1]
outputfile_directory = sys.argv[2]

#Creating a function that reads the video from the file
def read_video(input_filename):
    return cv2.VideoCapture(input_filename)

#Creating a function that captures and save the frame of the video
def save_frame(count, vid_cap, output_directory):
    #vid_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    #Reading the frame
    hasFrames, frame = vid_cap.read()

    if hasFrames:
        # if video is not finished, continue creating images
        name = os.path.join(output_directory, "frame" + str(count) + ".png")
        #writing the extracted images
        print(name)
        cv2.imwrite(name, frame)

    return hasFrames

def get_frames(input_filename, output_directory):
    #Capture images from a video at every (i.e, 5 sec, 5 mn, etc.) depending on the frame rate specified
    
    try:
        #Creating a folder
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    #Raise an error if folder is not created
    except OSError:
        logging.error('Error creating directory')

    #sec = 0
    count = 1 #Specify the number of images created
    videocap = read_video(input_filename)
    #success = save_frame(count, sec, videocap, output_directory)
    success = save_frame(count, videocap, output_directory)

    while success:
        count += 1
        #sec = sec + framerate
        #sec = round(sec, 2)
        success = save_frame(count, videocap, output_directory)
        #success = save_frame(count, sec, videocap, output_directory)

get_frames(input_filename, outputfile_directory)