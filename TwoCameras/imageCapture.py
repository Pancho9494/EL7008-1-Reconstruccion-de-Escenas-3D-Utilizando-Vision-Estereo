import cv2
import os
import glob


def capture(width, height, center):
    # Initialize capturing devices
    left = cv2.VideoCapture(1)
    left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    right = cv2.VideoCapture(2)
    right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Begin capture
    print("Begin capture")
    numCapture = 0
    while left.isOpened():
        successL, imgL = left.read()
        successR, imgR = right.read()

        key = cv2.waitKey(5)
        # Press Esc key to exit
        if key == 27:
            print("Pressed Esc, exiting...")
            break
        # Press s key to save picture
        elif key == ord('s'):
            print("Saving images")
            cv2.imwrite(f"calibrationImages/left/{numCapture}.png", imgL)
            cv2.imwrite(f"calibrationImages/right/{numCapture}.png", imgR)
            numCapture += 1
        # Press d key to empty folders
        elif key == ord('d'):
            print("Clearing folders")
            for file in glob.glob(f'calibrationImages/left/*'):
                os.remove(file)
            for file in glob.glob(f'calibrationImages/right/*'):
                os.remove(file)

        # Display cameras
        cv2.namedWindow('Left Camera')
        cv2.moveWindow('Left Camera', center[0], center[1])
        cv2.imshow('Left Camera', imgL)

        cv2.namedWindow('Right Camera')
        cv2.moveWindow('Right Camera', center[0] + width, center[1])
        cv2.imshow('Right Camera', imgR)


if __name__ == "__main__":
    capture(640, 480, (485, 170))
