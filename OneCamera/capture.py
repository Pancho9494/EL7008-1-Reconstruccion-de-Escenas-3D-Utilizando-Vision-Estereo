import cv2
import os
import glob


def capture(width, height):
    # Initialize capturing device
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if cam is None or not cam.isOpened():
        print("Unable to open video source")
        return

    # Begin capture
    print("Begin capture")
    numCapture = 0
    while cam.isOpened():
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
            if not cv2.imwrite(f"images/{numCapture}.png", img):
                raise Exception("Could not save image")
            numCapture += 1
        # Press d key to empty folders
        elif key == ord('d'):
            print("Clearing folder")
            numCapture = 0
            for file in glob.glob(f'./images/*'):
                os.remove(file)
        # Display camera
        cv2.namedWindow('Camera')
        cv2.imshow('Camera', img)


if __name__ == "__main__":
    capture(1280, 720)