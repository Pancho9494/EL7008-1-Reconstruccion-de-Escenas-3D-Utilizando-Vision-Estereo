import numpy as np
import cv2

def unDistort(width, height, center):
    # Load camera parameters
    stereoMapLx = np.load("./calibrationParams/stereoMapLx.npy")
    stereoMapLy = np.load("./calibrationParams/stereoMapLy.npy")
    stereoMapRx = np.load("./calibrationParams/stereoMapRx.npy")
    stereoMapRy = np.load("./calibrationParams/stereoMapRy.npy")

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
        key = cv2.waitKey(5)
        # Press Esc key to exit
        if key == 27:
            print("Pressed Esc, exiting...")
            break

        # Capture images
        successL, imgL = left.read()
        successR, imgR = right.read()

        # Undistort and images
        imgL = cv2.remap(imgL, stereoMapLx, stereoMapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        imgR = cv2.remap(imgR, stereoMapRx, stereoMapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Display cameras
        cv2.namedWindow('Left Camera')
        cv2.moveWindow('Left Camera', center[0], center[1])
        cv2.imshow('Left Camera', imgL)

        cv2.namedWindow('Right Camera')
        cv2.moveWindow('Right Camera', center[0] + width, center[1])
        cv2.imshow('Right Camera', imgR)


if __name__ == "__main__":
    unDistort(640, 480, (485, 170))