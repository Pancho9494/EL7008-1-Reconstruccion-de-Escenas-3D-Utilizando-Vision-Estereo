import numpy as np
from tqdm import tqdm
import glob
import cv2


def getPoints(boardSize, verbose=False):
    objP = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
    objP[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)

    objPoints = []  # 3D points in real space
    imgPoints = []  # 2D points in camera plane

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    iterable = glob.glob("./images/*")
    if not verbose:
        iterable = tqdm(iterable)
    for path in iterable:
        image = cv2.imread(path, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # does removing this line improve performance?
        # (thresh, bin) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        ret, corners = cv2.findChessboardCorners(gray, boardSize, None)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
            objPoints.append(objP)
            imgPoints.append(corners)

            if verbose:
                print("Found corners")
                cv2.drawChessboardCorners(image, boardSize, corners, ret)
                cv2.namedWindow('Camera')
                cv2.imshow("Camera", image)
                cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return objPoints, imgPoints


def calibrate(objPoints, imgPoints, resolution):
    ret, CM, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, resolution, None, None)
    return ret, CM, dist, rvecs, tvecs


if __name__ == "__main__":
    # (9, 6) size of calibration chess board
    objPoints, imgPoints = getPoints((9, 6), verbose=False)
    ret, CM, dist, rvecs, tvecs = calibrate(objPoints, imgPoints, (1280, 720))
    np.save("params/ret", ret)
    np.save("params/CM", CM)
    np.save("params/dist", dist)
    np.save("params/rvecs", rvecs)
    np.save("params/tvecs", tvecs)
    print(f"Re-projection error = {ret}")

