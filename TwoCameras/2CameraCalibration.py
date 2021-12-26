import numpy as np
import cv2
import glob
from tqdm import tqdm


error = (0,0)


# squareLen = 25  # Length of chessboard squares in mm
def getPoints(boardSize, verbose=False):
    objP = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
    objP[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    # objP = objP * squareLen # necessary?

    objPoints = []  # 3D points in real space
    imgPointsL = []  # 2D points in left camera plane
    imgPointsR = []  # 2D points in right camera plane

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    iterable = zip(glob.glob("calibrationImages/left/*.png"), glob.glob('calibrationImages/right/*.png'))
    if not verbose:
        iterable = tqdm(iterable)

    for nameL, nameR in iterable:
        # Read images and convert them to grayscale
        imL = cv2.imread(nameL)
        grayL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        # (threshL, binL) = cv2.threshold(grayL, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        imR = cv2.imread(nameR)
        grayR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
        # (threshR, binR) = cv2.threshold(grayR, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, boardSize, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, boardSize, None)

        # If there are corners, draw points
        if retL and retR:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

            objPoints.append(objP)
            imgPointsL.append(cornersL)
            imgPointsR.append(cornersR)

            if verbose:
                print("Found corners")
                # Draw and display the corners
                cv2.drawChessboardCorners(imL, boardSize, cornersL, retL)
                cv2.namedWindow('Left Camera')
                # cv2.moveWindow('Left Camera', center[0], center[1])
                cv2.imshow('Left Camera', imL)

                cv2.drawChessboardCorners(imR, boardSize, cornersR, retR)
                cv2.namedWindow('Right Camera')
                # cv2.moveWindow('Right Camera', center[0] + width, center[1])
                cv2.imshow('Right Camera', imR)
                cv2.waitKey(2000)

    cv2.destroyAllWindows()
    return objPoints, imgPointsL, imgPointsR


def calibrate(points, pointsL, pointsR, resolution):
    retL, CML, distL, rvecsL, tvecsL = cv2.calibrateCamera(points, pointsL, resolution, None, None)
    newCML, roiL = cv2.getOptimalNewCameraMatrix(CML, distL, resolution, 1)

    retR, CMR, distR, rvecsR, tvecsR = cv2.calibrateCamera(points, pointsR, resolution, None, None)
    newCMR, roiR = cv2.getOptimalNewCameraMatrix(CMR, distR, resolution, 1)

    np.save("calibrationParams/roiL", roiL)
    np.save("calibrationParams/roiR", roiR)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    return cv2.stereoCalibrate(points, pointsL, pointsR, newCML, distL, newCMR, distR, resolution, criteria, flags)


def rectification(newCML, newCMR, resolution, rot, trans, scale):
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv2.stereoRectify(newCML, distL, newCMR, distR,
                                                                                resolution, rot, trans, scale,
                                                                                (0, 0))
    stereoMapL = cv2.initUndistortRectifyMap(newCML, distL, rectL, projMatrixL, resolution, cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCMR, distR, rectR, projMatrixR, resolution, cv2.CV_16SC2)
    return stereoMapL, stereoMapR




if __name__ == "__main__":
    objPoints, imgPointsL, imgPointsR = getPoints((9, 6), False)
    retStereo, newCML, distL, newCMR, distR, rot, trans, EMatrix, FMatrix = calibrate(objPoints,
                                                                                      imgPointsL,
                                                                                      imgPointsR,
                                                                                      (1280, 720))
    stereoMapL, stereoMapR = rectification(newCML, newCMR, (1280, 720), rot, trans, 1)
    np.save("calibrationParams/stereoMapLx", stereoMapL[0])
    np.save("calibrationParams/stereoMapLy", stereoMapL[1])
    np.save("calibrationParams/stereoMapRx", stereoMapR[0])
    np.save("calibrationParams/stereoMapRy", stereoMapR[1])
    print(f"Reprojection error: Left = {error[0]} \t Right = {error[1]}")
    print(f"Reprojection stereo error = {retStereo}")