import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def capture(width, height):
    names = ["L", "R"]
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if cam is None or not cam.isOpened():
        print("Unable to open video source")
        return

    numCapture = 0
    while numCapture < 2:
        successL, img = cam.read()
        key = cv2.waitKey(5)
        # Press Esc key to exit
        if key == 27:
            print("Pressed Esc, exiting...")
            break
        # Press s key to save picture
        elif key == ord('s'):
            print("Saving images")
            print(numCapture)
            if not cv2.imwrite(f"reconstructionImages/{names[numCapture]}.png", img):
                raise Exception("Could not save image")
            numCapture += 1
        # Press d key to empty folders
        elif key == ord('d'):
            print("Clearing folder")
            numCapture = 0
            for file in glob.glob(f'reconstructionImages/*'):
                os.remove(file)
        # Display camera
        cv2.namedWindow('Camera')
        cv2.imshow('Camera', img)
    cv2.destroyAllWindows()
    return cv2.imread("reconstructionImages/R.png"), cv2.imread("reconstructionImages/L.png")


def SGBM(imgL, imgR, winSize, minDisp, maxDisp):
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
    resolution = (1280, 720)
    ret = np.load('params/ret.npy')
    CM = np.load('params/CM.npy')
    dist = np.load('params/dist.npy')

    imL, imR = capture(resolution[0], resolution[1])

    newCM, roi = cv2.getOptimalNewCameraMatrix(CM, dist, resolution, 1)

    imL = cv2.undistort(imL, CM, dist, None, newCM)
    imR = cv2.undistort(imR, CM, dist, None, newCM)

    imL = cv2.resize(imL, (int(imL.shape[1] / 3), int(imL.shape[0] / 3)))
    imR = cv2.resize(imR, (int(imR.shape[1] / 3), int(imR.shape[0] / 3)))
    cv2.imwrite("reconstructionImages/Ldown.png", imL)
    cv2.imwrite("reconstructionImages/Rdown.png", imR)
    print(imL.shape, imR.shape)

    disparityMap = SGBM(imgL=imR, imgR=imL, winSize=5, minDisp=-1, maxDisp=61)
    np.save("params/disparityMap", disparityMap)
