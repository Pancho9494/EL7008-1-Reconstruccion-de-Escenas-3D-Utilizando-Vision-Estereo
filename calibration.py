import numpy as np
import cv2
import glob


# squareLen = 25  # Length of chessboard squares in mm
def getPoints(width, height, center, boardSize, squareLen):
    objP = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
    objP[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    objP = objP * squareLen

    objPoints = []  # 3D points in real space
    imgPointsL = []  # 2D points in left camera plane
    imgPointsR = []  # 2D points in right camera plane

    for nameL, nameR in zip(glob.glob('images/left/*.png'), glob.glob('images/right/*.png')):
        # Read images and convert them to grayscale
        imL = cv2.imread(nameL)
        grayL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        (threshL, binL) = cv2.threshold(grayL, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        imR = cv2.imread(nameR)
        grayR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
        (threshR, binR) = cv2.threshold(grayR, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find board corners
        retL, cornersL = cv2.findChessboardCorners(binL, boardSize, None)
        retR, cornersR = cv2.findChessboardCorners(binR, boardSize, None)

        # If there are corners, draw points
        if retL and retR:
            print("Found corners")
            objPoints.append(objP)

            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgPointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgPointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(imL, boardSize, cornersL, retL)
            cv2.namedWindow('Left Camera')
            cv2.moveWindow('Left Camera', center[0], center[1])
            cv2.imshow('Left Camera', imL)

            cv2.drawChessboardCorners(imR, boardSize, cornersR, retR)
            cv2.namedWindow('Right Camera')
            cv2.moveWindow('Right Camera', center[0] + width, center[1])
            cv2.imshow('Right Camera', imL)
            cv2.waitKey(2000)

    cv2.destroyAllWindows()
    return objPoints, imgPointsL, imgPointsR


def calibrate(points, pointsL, pointsR, resolution):
    retL, CML, distL, rvecsL, tvecsL = cv2.calibrateCamera(points, pointsL, resolution, None, None)
    newCML, roiL = cv2.getOptimalNewCameraMatrix(CML, distL, resolution, 1)

    retR, CMR, distR, rvecsR, tvecsR = cv2.calibrateCamera(points, pointsR, resolution, None, None)
    newCMR, roiR = cv2.getOptimalNewCameraMatrix(CMR, distR, resolution, 1)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    return cv2.stereoCalibrate(points, pointsL, pointsR, newCML, distL, newCMR, distR, resolution, criteria, flags)


def rectification(newCML, newCMR, resolution, rot, trans, scale):
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv2.stereoRectify(newCML, distL, newCMR, distR,
                                                                                resolution, rot, trans, scale,
                                                                                (0, 0))
    stereoMapL = cv2.initUndistortRectifyMap(newCML, distL, rectL, projMatrixL, resolution, cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCMR, distR, rectR, projMatrixR, resolution, cv2.CV_16SC2)
    return stereoMapL, stereoMapR




if __name__ == "__main__":
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    objPoints, imgPointsL, imgPointsR = getPoints(640, 480, (485, 170), (7, 7), 25)
    retStereo, newCML, distL, newCMR, distR, rot, trans, EMatrix, FMatrix = calibrate(objPoints,
                                                                                      imgPointsL,
                                                                                      imgPointsR,
                                                                                      (640, 480))
    stereoMapL, stereoMapR = rectification(newCML, newCMR, (640, 480), rot, trans, 1)
    np.save("./calibrationParams/stereoMapLx", stereoMapL[0])
    np.save("./calibrationParams/stereoMapLy", stereoMapL[1])
    np.save("./calibrationParams/stereoMapRx", stereoMapR[0])
    np.save("./calibrationParams/stereoMapRy", stereoMapR[1])