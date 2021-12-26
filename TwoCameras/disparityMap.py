import matplotlib.pyplot as plt
import cv2
from unDistortedCapture import *

class StereoBMTuner(object):
    """
    A class for tuning Stereo BM settings.

    Display a normalized disparity picture from two pictures captured with a
    ``CalibratedPair`` and allow the user to manually tune the settings for the
    stereo block matcher.
    """
    #: Window to show results in
    window_name = "Stereo BM Tuner"

    def __init__(self, calibrated_pair, image_pair):
        """Initialize tuner with a ``CalibratedPair`` and tune given pair."""
        #: Calibrated stereo pair to find Stereo BM settings for
        self.calibrated_pair = calibrated_pair
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("cam_preset", self.window_name,
                           self.calibrated_pair.stereo_bm_preset, 3,
                           self.set_bm_preset)
        cv2.createTrackbar("ndis", self.window_name,
                           self.calibrated_pair.search_range, 160,
                           self.set_search_range)
        cv2.createTrackbar("winsize", self.window_name,
                           self.calibrated_pair.window_size, 21,
                           self.set_window_size)
        #: (left, right) image pair to find disparity between
        self.pair = image_pair
        self.tune_pair(image_pair)

    def set_bm_preset(self, preset):
        """Set ``search_range`` and update disparity image."""
        try:
            self.calibrated_pair.stereo_bm_preset = preset
        except:
            return
        self.update_disparity_map()

    def set_search_range(self, search_range):
        """Set ``search_range`` and update disparity image."""
        try:
            self.calibrated_pair.search_range = search_range
        except:
            return
        self.update_disparity_map()

    def set_window_size(self, window_size):
        """Set ``window_size`` and update disparity image."""
        try:
            self.calibrated_pair.window_size = window_size
        except:
            return
        self.update_disparity_map()

    def update_disparity_map(self):
        """Update disparity map in GUI."""
        disparity = self.calibrated_pair.compute_disparity(self.pair)
        cv2.imshow(self.window_name, disparity / 255.)
        cv2.waitKey()

    def tune_pair(self, pair):
        """Tune a pair of images."""
        self.pair = pair
        self.update_disparity_map()


def SGBM(winSize, minDisp, maxDisp):
    imgL = cv2.imread("reconstructionImages/left/0.png")
    imgR = cv2.imread("reconstructionImages/right/0.png")
    numDisp = maxDisp - minDisp

    stereo = cv2.StereoSGBM_create(minDisparity=minDisp,
                                   numDisparities=numDisp,
                                   blockSize=11,
                                   uniquenessRatio=5,
                                   speckleWindowSize=50,
                                   speckleRange=1,
                                   disp12MaxDiff=16,
                                   P1=8 * 3 * winSize ** 2,
                                   P2=32 * 3 * winSize ** 2)
    disparityMap = stereo.compute(imgL, imgR)
    plt.imshow(disparityMap, 'gray')
    plt.show()
    return disparityMap


if __name__ == "__main__":
    # disparityMap = SGBM(5, -1, 61)
    unDistort(1280,720)