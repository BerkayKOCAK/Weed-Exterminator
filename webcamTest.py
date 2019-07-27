import cv2
import argparse
import numpy as np
import time

# inherit camera to var.
cam = cv2.VideoCapture(0);
while(cam.isOpened()):

    # read inputs
    check, frame = cam.read() 
    #print(check);
    #print(frame);

    #show frame
    cv2.imshow("Capturing ! ", frame);

#shutdown
cam.release();