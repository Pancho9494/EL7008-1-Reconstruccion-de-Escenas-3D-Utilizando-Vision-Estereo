import numpy as np
import cv2
import matplotlib.pyplot as plt


def SGBM(imgL, imgR, minDisp, maxDisp, blockSize, uniqRatio, speckWin):
    numDisp = maxDisp - minDisp

    stereo = cv2.StereoSGBM_create(minDisparity=minDisp,
                                   numDisparities=numDisp,
                                   blockSize=blockSize,
                                   uniquenessRatio=uniqRatio,
                                   speckleWindowSize=speckWin,
                                   speckleRange=1,
                                   disp12MaxDiff=16,
                                   P1=8 * 3 * blockSize ** 2,
                                   P2=32 * 3 * blockSize ** 2)
    disparityMap = stereo.compute(imgL, imgR)
    # plt.imshow(disparityMap, 'gray')
    # plt.show()
    return disparityMap


def unDistort(width, height):
    # Load camera parameters
    stereoMapLx = np.load("calibrationParams/stereoMapLx.npy")
    stereoMapLy = np.load("calibrationParams/stereoMapLy.npy")
    stereoMapRx = np.load("calibrationParams/stereoMapRx.npy")
    stereoMapRy = np.load("calibrationParams/stereoMapRy.npy")
    roiL = np.load("calibrationParams/roiL.npy")
    roiR = np.load("calibrationParams/roiR.npy")

    # Initialize capturing devices
    left = cv2.VideoCapture(0)
    left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    right = cv2.VideoCapture(2)
    right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if left is None or not left.isOpened():
        print("Unable to open video source")
        return

    # Begin capture
    print("Begin capture")
    numCapture = 0
    blockSize = uniqRatio = speckWin = 0
    cv2.namedWindow('Disparity Map')
    cv2.createTrackbar("block size", "Disparity Map", -1, 100, foo)
    cv2.createTrackbar("uniqueness ratio", "Disparity Map", 5, 15, foo)
    cv2.createTrackbar("speckle window size", "Disparity Map", 50, 200, foo)
    while left.isOpened():
        blockSize = cv2.getTrackbarPos("block size", "Disparity Map")
        uniqRatio = cv2.getTrackbarPos("uniqueness ratio", "Disparity Map")
        speckWin = cv2.getTrackbarPos("speckle window size", "Disparity Map")

        # Capture images
        successL, captureL = left.read()
        successR, captureR = right.read()

        # Un-distort images
        imgL = cv2.remap(captureL, stereoMapLx, stereoMapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        imgR = cv2.remap(captureR, stereoMapRx, stereoMapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Down-sample images
        imgL = cv2.resize(imgL, (int(imgL.shape[1] / 2), int(imgL.shape[0] / 2)))
        imgR = cv2.resize(imgR, (int(imgR.shape[1] / 2), int(imgR.shape[0] / 2)))

        # Crop images
        # xL, yL, wL, hL = roiL
        # xR, yR, wR, hR = roiR
        # imgL = imgL[yL:yL + hL, xL:xL + wL]
        # imgR = imgR[yR:yR + hR, xR:xR + wR]

        # Compute disparity map
        disparityMap = SGBM(imgL=imgL, imgR=imgR, minDisp=-1, maxDisp=63, blockSize=blockSize, uniqRatio=uniqRatio, speckWin=speckWin) / 255

        key = cv2.waitKey(5)
        # Press Esc key to exit
        if key == 27:
            print("Pressed Esc, exiting...")
            break
        # Press s key to save picture
        # elif key == ord('s'):
        #     print("Saving images")
        #     cv2.imwrite(f"reconstructionImages/left/{numCapture}.png", imgL)
        #     cv2.imwrite(f"reconstructionImages/right/{numCapture}.png", imgR)

        # Display cameras
        cv2.imshow('Disparity Map', disparityMap)

        cv2.namedWindow('Left Camera')
        cv2.imshow('Left Camera', imgL)

        cv2.namedWindow('Right Camera')
        cv2.imshow('Right Camera', imgR)
    left.release()
    right.release()
    return imgL, imgR


def foo(x):
    print(x)


if __name__ == "__main__":
    imgL, imgR = unDistort(1280, 720)
