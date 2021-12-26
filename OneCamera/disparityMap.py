import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


def unDistortedCapture(width, height, CM, dist, newCM):
    names = ["L", "R"]
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if cam is None or not cam.isOpened():
        print("Unable to open video source")
        return

    numCapture = 0
    while numCapture < 2:
        successL, img = cam.read()
        img = cv2.undistort(img, CM, dist, None, newCM)

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
    return cv2.imread("reconstructionImages/L.png"), cv2.imread("reconstructionImages/R.png")


# mode 0 for BM
# mode 1 for SGBM
def computeDisparity(imgL, imgR, winSize, numDisparity, mode, applyFilter=True):
    # Ensure images are in grayscale
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Choose BM or SGBM as matcher
    info = f"{numDisparity} {winSize}"
    if mode == "sgbm":
        uniquenessRatio = 5
        speckleWindow = 100
        speckleRange = 1
        disp12 = 16
        info += f" {uniquenessRatio} {speckleWindow} {speckleRange} {disp12}"
        leftMatcher = cv2.StereoSGBM.create(minDisparity=-1,
                                            numDisparities=numDisparity,
                                            blockSize=winSize,
                                            uniquenessRatio=uniquenessRatio,
                                            speckleWindowSize=speckleWindow,
                                            speckleRange=speckleRange,
                                            disp12MaxDiff=disp12,
                                            P1=8 * 3 * winSize ** 2,
                                            P2=32 * 3 * winSize ** 2)
    elif mode == "bm":
        leftMatcher = cv2.StereoBM.create(numDisparities=numDisparity,
                                          blockSize=winSize)

    # Compute left pov disparity
    disparityLeft = leftMatcher.compute(imgL, imgR)

    # Apply weighted least squares filter
    out = disparityLeft
    if applyFilter:
        # parameters of post-filtering
        lmbda = 8000.0
        sigma = 2.5

        # matcher for right pov
        rightMatcher = cv2.ximgproc.createRightMatcher(leftMatcher)

        disparityRight = rightMatcher.compute(imgR, imgL)

        # filter instance
        WLSFilter = cv2.ximgproc.createDisparityWLSFilter(leftMatcher)
        WLSFilter.setLambda(lmbda)
        WLSFilter.setSigmaColor(sigma)
        out = WLSFilter.filter(disparityLeft, imgL, disparity_map_right=disparityRight)
        info += f" | λ: {lmbda} σ: {sigma}"

    # Visualize result
    plt.imshow(out, 'gray')
    plt.title(info)
    plt.show()
    return out


# readability function
def resize(img, downRatio):
    return cv2.resize(img, (int(img.shape[1] / downRatio), int(img.shape[0] / downRatio)))


if __name__ == "__main__":
    resolution = (1280, 720)

    # load calibration parameters
    ret = np.load('params/ret.npy')
    CM = np.load('params/CM.npy')
    dist = np.load('params/dist.npy')

    newCM, roi = cv2.getOptimalNewCameraMatrix(CM, dist, resolution, 1)

    # capture undistorted images and resize
    imL, imR = unDistortedCapture(resolution[0], resolution[1], CM, dist, newCM)
    imL = resize(imL, 3)
    imR = resize(imR, 3)

    # compute and save disparity
    disparityMap = computeDisparity(imgL=imL, imgR=imR, winSize=11, numDisparity=128, mode="bm", applyFilter=False)
    np.save("params/disparityMap", disparityMap)
