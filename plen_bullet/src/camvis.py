#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:20:17 2019

@author: mori
"""
import cv2
import time
import numpy as np


def vision(radius, centre, xmid, ymid):
    ret, frame = cap.read(
    )  #ret is True or False (connected or not). Frame is the next frame from the camera using cap.read()

    # Our operations on the frame come here
    # convert frame to HSV cspace
    blurred = cv2.GaussianBlur(frame, (11, 11),
                               0)  #(,) denotes gaussian kernel size
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # make a mask for the color "red, perform erosions to remove noise and dialations to resize target to original
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.dilate(mask, None, iterations=10)

    # find mask contours and initialize and ball centroid
    cnts, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)  # What does RETR_EXTERNAL do?
    #RETR_EXTERNAL only find outer/most extreme contours

    #now sort contours by size
    if len(list(cnts)) > 0:
        cnt_max = max(cnts, key=cv2.contourArea)

        per_max = cv2.arcLength(cnt_max, True)

        contour_area = cv2.contourArea(cnt_max)

        circularity = contour_area / (per_max**2)
        #print(circularity)

        #if (1/(4.0 *np.pi))*0.5 <= circularity and circularity <= (1/(4.0*np.pi))*1:
        #print('circle!')

        ((x, y), radius) = cv2.minEnclosingCircle(
            cnt_max)  #returns min enclosing circle radius and its centroid

        M = cv2.moments(cnt_max)
        centre = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    else:
        centre = (xmid, ymid)
        radius = radius

        # only proceed if the radius meets a minimum size
    radius = int(radius)
    if radius > 50:
        cv2.circle(frame, centre, radius, (0, 0, 255),
                   2)  #-1 is full circle, 2 is outline
    #print(radius)

    if len(list(cnts)) > 0:
        cv2.line(frame, (centre[0], 0), (centre[0], 480), (255, 0, 0), 2)
        cv2.line(frame, (0, centre[1]), (640, centre[1]), (0, 255, 0), 2)

    vision = [radius, centre, frame]
    return vision


cap = cv2.VideoCapture(0)

redLower = (10, 141, 131)  # 0 0 129
redUpper = (23, 255, 255)  # 78 255 255
radius = 1
ymid = 480 / 2
xmid = 640 / 2
centre = (xmid, ymid)
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 400)
fontScale = 1
fontColor = (255, 0, 0)
lineType = 2
while (True):

    start_time = time.time()  #used to time loop
    # Capture frame-by-frame

    #VISION
    vision_param = vision(radius, centre, xmid, ymid)
    radius = vision_param[0]
    centre = vision_param[1]
    frame = vision_param[2]

    #now find angles relative to camera centre
    #650x500
    F = 1946.15  #camera focal length
    real_rad = 1.3  #inches
    obj_dist = (real_rad * F) / radius
    x_px = centre[0] - xmid
    y_px = ymid - centre[1]

    #note that radius_px = radius_inch
    x_dist = real_rad * x_px / radius
    y_dist = real_rad * y_px / radius

    #print(y_dist)

    pancam = np.degrees(np.arctan(x_dist / obj_dist))
    tiltcam = np.degrees(np.arctan(y_dist / obj_dist))

    # print(tiltcam)

    #print('tilt:', tiltcam)
    #print('pan:', pancam)
    #print fps
    FPS = cap.get(cv2.CAP_PROP_FPS)  #camera fps
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(frame, 'FPS:', FPS, (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    #print("FPS:", FPS)

    #print loop time
    end_time = time.time()

    looptime = end_time - start_time

    cv2.putText(frame, str(1.0 / looptime), bottomLeftCornerOfText, font, fontScale,
                fontColor, lineType)

    # show img
    cv2.imshow('frame', frame)

    #print("LPS:", 1/looptime)

    #quit program
    if cv2.waitKey(1) & 0xFF == ord(
            'q'
    ):  # 0xFF is 8bit hex mask which is passed to cv2.waitKey
        # so if key = q, then quit
        break

# When everything done, release the capture
cap.release()  #disconnect from camera
cv2.destroyAllWindows()  #close camera windows (imshow)