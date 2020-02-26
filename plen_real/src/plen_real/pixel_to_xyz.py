import numpy as np
import cv2
from plen_real.object_detector import ObjectDetector
import os


class PixelToXYZ:

    # camera variables
    cam_matrix = None
    dist = None
    newcam_matrix = None
    roi = None
    rvec1 = None
    tvec1 = None
    R_matrix = None
    Rt = None
    P_matrix = None

    # images
    img = None

    def __init__(self):

        # Find abs path to this file
        my_path = os.path.abspath(os.path.dirname(__file__))
        captures_path = os.path.join(my_path, "../captures")
        camera_data_path = os.path.join(my_path, "../camera_data")

        if not os.path.exists(captures_path):
            os.makedirs(captures_path)

        if not os.path.exists(camera_data_path):
            os.makedirs(camera_data_path)

        self.obj_detect = ObjectDetector()

        # self.obj_detect=image_recognition_singlecam.image_recognition(True,False,captures_path,captures_path,True,True)

        self.cam_matrix = np.load(camera_data_path + 'cam_matrix.npy')
        self.dist = np.load(camera_data_path + 'dist.npy')
        self.newcam_matrix = np.load(camera_data_path + 'newcam_matrix.npy')
        self.roi = np.load(camera_data_path + 'roi.npy')
        self.rvec1 = np.load(camera_data_path + 'rvec1.npy')
        self.tvec1 = np.load(camera_data_path + 'tvec1.npy')
        self.R_matrix = np.load(camera_data_path + 'R_matrix.npy')
        self.Rt = np.load(camera_data_path + 'Rt.npy')
        self.P_matrix = np.load(camera_data_path + 'P_matrix.npy')

        s_arr = np.load(camera_data_path + 's_arr.npy')
        self.scalingfactor = s_arr[0]

        self.inverse_newcam_matrix = np.linalg.inv(self.newcam_matrix)
        self.inverse_R_matrix = np.linalg.inv(self.R_matrix)

    def previewImage(self, text, img):
        # show full screen
        cv2.namedWindow(text, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(text, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        cv2.imshow(text, img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    def undistort_image(self, image):
        image_undst = cv2.undistort(image, self.cam_matrix, self.dist, None,
                                    self.newcam_matrix)

        return image_undst

    def load_background(self, background):
        self.bg_undst = self.undistort_image(background)
        self.bg = background

    def detect_xyz(self, image, calcXYZ=True, calcarea=False):

        image_src = image.copy()

        # if calcXYZ==True:
        #    img= self.undistort_image(image_src)
        #    bg = self.bg_undst
        # else:
        img = image_src

        XYZ = []
        # self.previewImage("capture image",img_undst)
        # self.previewImage("bg image",self.bg_undst)
        vision_param = self.obj_detect.detect()

        radius = vision_param[0]
        centre = vision_param[1]
        frame = vision_param[2]

        return img, XYZ

    def calculate_XYZ(self, u, v):

        # Solve: From Image Pixels, find World Points

        uv_1 = np.array([[u, v, 1]], dtype=np.float32)
        uv_1 = uv_1.T
        suv_1 = self.scalingfactor * uv_1
        xyz_c = self.inverse_newcam_matrix.dot(suv_1)
        xyz_c = xyz_c - self.tvec1
        XYZ = self.inverse_R_matrix.dot(xyz_c)

        return XYZ