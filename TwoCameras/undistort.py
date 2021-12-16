import numpy as np
import cv2

def capture(width, height):
    imgL = None
    imgR = None
    # Initialize capturing devices
    left = cv2.VideoCapture(1)
    left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    right = cv2.VideoCapture(2)
    right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Begin capture
    print("Begin capture")
    while left.isOpened():
        # Capture images
        successL, captureL = left.read()
        successR, captureR = right.read()

        key = cv2.waitKey(5)
        # Press Esc key to exit
        if key == 27:
            print("Pressed Esc, exiting...")
            break
        # Press s key to save picture
        elif key == ord('s'):
            print("Saving images")
            imgL = captureL
            imgR = captureR
    left.release()
    right.release()
    return imgL, imgR


def unDistort(width, height):
    # Load camera parameters
    stereoMapLx = np.load("calibrationParams/stereoMapLx.npy")
    stereoMapLy = np.load("calibrationParams/stereoMapLy.npy")
    stereoMapRx = np.load("calibrationParams/stereoMapRx.npy")
    stereoMapRy = np.load("calibrationParams/stereoMapRy.npy")

    # Capture images
    imgL, imgR = capture(width, height)

    # Un-distort images
    imgL = cv2.remap(imgL, stereoMapLx, stereoMapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    imgR = cv2.remap(imgR, stereoMapRx, stereoMapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Down-sample images
    imgL = cv2.resize(imgL, (int(imgL.shape[0]/3), int(imgL.shape[1]/3)))
    imgR = cv2.resize(imgR, (int(imgR.shape[0]/3), int(imgR.shape[1]/3)))
    return imgL, imgR


# Semi Global Block Matching
# 1. Prefilter images to normalize brightness and enhance texture
# 2. Correspondence search along horizontal epipolar lines using SAD windows
# 3. Post-filtering to eliminate bad correspondence matches
def SGBM():
    return




if __name__ == "__main__":
    imgL, imgR = unDistort(640, 480)

