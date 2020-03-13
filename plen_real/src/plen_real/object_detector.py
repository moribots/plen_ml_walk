#!/usr/bin/env python
import cv2
import numpy as np
import time


class ObjectDetector:
    def __init__(self,
                 thresh_lower=(10, 141, 131),
                 thresh_upper=(23, 255, 255),
                 radius=0,
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
        self.cap = cv2.VideoCapture(self.cam)
        # self.cap.set(cv2.CV_CAP_PROP_FPS, 60)

        self.x_dist = 1.43  # meters
        self.y_dist = 1.04  # meters
        self.z_dist = 1.89  # meters

        self.pixel_x = self.x_dist / 640.0
        self.pixel_y = self.y_dist / 480

        self.x_position = 0
        self.y_position = 0
        self.z_position = 0

    def detect(self):
        """ Detect the object's position in pixel coordinates
        """
        cap = self.cap
        ret, frame = cap.read()

        # print(frame.shape)
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
        # cv2.imshow('processed', mask)

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

            # PRINT X,Y POSITION
            # Global Zero x: 0.208
            # Global Zero y: 0.3940
            self.x_position = (640 - centre[0]) * self.pixel_x - 0.208
            print("X Position: {}".format(self.x_position))
            self.y_position = centre[1] * self.pixel_y - 0.394
            print("Y Position: {}".format(self.y_position))
            # 13px Radius at 0.020m height
            # 20px at 0.079m height
            # actual radius is 0.0225 m
            m = (0.079 - 0.02) / (20 - 13)
            b = 0.079 - 20 * m
            self.z_position = radius * m + b
            print("Z Position: {}".format(self.z_position))


            print("RADIUS: {}".format(radius))

        vision = [radius, centre, frame]
        return vision

    def show(self):

        while (True):

            start_time = time.time()

            vision_param = self.detect()

            end_time = time.time()

            print("FPS: {}".format(1.0 / (end_time - start_time)))
            # radius = vision_param[0]
            # centre = vision_param[1]
            frame = vision_param[2]

            # show img
            cv2.imshow('frame', frame)

            # quit program
            # NOTE: value in waitKey is in ms, this is how long we wait before
            # calling loop again
            if cv2.waitKey(16) == 27:
                break

        # When everything done, release the capture
        cap.release()  # disconnect from camera
        cv2.destroyAllWindows()  # close camera windows (imshow)


if __name__ == "__main__":
    obj = ObjectDetector()
    obj.show()