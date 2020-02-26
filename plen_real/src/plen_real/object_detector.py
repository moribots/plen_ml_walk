#!/usr/bin/env python
import cv2
import numpy as np
import time


class ObjectDetector:
    def __init__(self,
                 thresh_lower=(10, 141, 131),
                 thresh_upper=(23, 255, 255),
                 radius=1,
                 min_radius=2,
                 ymid=480 / 2.0,
                 xmid=640 / 2.0):

        self.thresh_lower = (92, 46, 25)  # purple
        self.thresh_upper = (167, 163, 255)  # purple
        self.radius = radius
        self.min_radius = min_radius
        self.ymid = ymid
        self.xmid = xmid
        self.centre = (xmid, ymid)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, 400)
        self.fontScale = 1
        self.fontColor = (255, 0, 0)
        self.lineType = 2
        self.cam = 0
        self.iterations = 5

    def detect(self, cap):
        """ Detect the object's position in pixel coordinates
        """
        ret, frame = cap.read()
        # ret is True or False (connected or not).
        # frame is the next frame from the camera using cap.read()

        # Our operations on the frame come here
        # convert frame to HSV cspace
        # (,) denotes gaussian kernel size
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # make a mask for the desired color and threshold the image
        mask = cv2.inRange(hsv, self.thresh_lower, self.thresh_upper)
        # Perform erosions to remove noise
        mask = cv2.erode(mask, None, iterations=self.iterations)
        # Perform dialations to restore the eroded shape
        mask = cv2.dilate(mask, None, iterations=self.iterations)
        cv2.imshow('processed', mask)

        # find mask contours
        cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        # RETR_EXTERNAL only find outer/most extreme contours

        # now sort contours by size
        if len(list(cnts)) > 0:
            # Retrieve the largest captured contour
            cnt_max = max(cnts, key=cv2.contourArea)

            # Retrieve the largest captured perimeter
            # per_max = cv2.arcLength(cnt_max, True)

            # Extract the largest contour's area
            # contour_area = cv2.contourArea(cnt_max)

            # circularity = contour_area / (per_max**2)
            # print(circularity)

            # if (1/(4.0 *np.pi))*0.5 <= circularity and circularity <= (1/(4.0*np.pi))*1:
            # print('circle!')

            # returns min enclosing circle radius and its centroid
            ((x, y), radius) = cv2.minEnclosingCircle(cnt_max)

            M = cv2.moments(cnt_max)
            centre = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        else:
            centre = (self.xmid, self.ymid)
            radius = self.radius

            # only proceed if the radius meets a minimum size
        radius = int(radius)
        if radius > self.min_radius:
            cv2.circle(frame, centre, radius, (0, 140, 255),
                       2)  # -1 is full circle, 2 is outline
            # print(radius)

        if len(list(cnts)) > 0:
            # Draw Crosshairs
            cv2.line(frame, (centre[0], 0), (centre[0], 480), (0, 140, 255), 2)
            cv2.line(frame, (0, centre[1]), (640, centre[1]), (0, 140, 255), 2)

        vision = [radius, centre, frame]
        return vision

    def show(self):

        cap = cv2.VideoCapture(self.cam)

        while (True):

            vision_param = self.detect(cap)
            # radius = vision_param[0]
            # centre = vision_param[1]
            frame = vision_param[2]

            # show img
            cv2.imshow('frame', frame)

            # quit program
            if cv2.waitKey(1) & 0xFF == ord(
                    'q'
            ):  # 0xFF is 8bit hex mask which is passed to cv2.waitKey
                # so if key = q, then quit
                break

        # When everything done, release the capture
        cap.release()  # disconnect from camera
        cv2.destroyAllWindows()  # close camera windows (imshow)


if __name__ == "__main__":
    obj = ObjectDetector()
    obj.show()