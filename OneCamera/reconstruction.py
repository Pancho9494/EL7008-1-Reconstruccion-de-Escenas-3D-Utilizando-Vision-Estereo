import numpy as np
import cv2


# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


#
def increaseBrightness(img, value=30):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(HSV)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    finalHSV = cv2.merge((h, s, v))
    img = cv2.cvtColor(finalHSV, cv2.COLOR_HSV2BGR)
    return img


def reconstruct(resolution):
    disparityMap = np.load("params/disparityMap.npy")
    CM = np.load("params/CM.npy")
    (w, h) = (int(resolution[0]), int(resolution[1]))
    focalLength = CM[0][0] / CM[1][1]

    # Compute perspective transformation matrix
    Q = np.float32([[1, 0, 0, -w / 2.0],
                    [0, -1, 0, h / 2.0],
                    [0, 0, 0, -focalLength],
                    [0, 0, 1, 0]])
    Q2 = np.float32([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, focalLength * 0.05, 0],
                     [0, 0, 0, 1]])

    # Reconstruct
    imL = cv2.imread("reconstructionImages/Ldown.png")
    imL = increaseBrightness(imL, value=30)

    points3D = cv2.reprojectImageTo3D(disparityMap, Q2)
    colors = cv2.cvtColor(imL, cv2.COLOR_BGR2RGB)

    # Get rid of points with value 0 (i.e no depth)
    maskMap = disparityMap > disparityMap.min()

    # Mask colors and points.
    outputPoints = points3D[maskMap]
    outputColors = colors[maskMap]

    create_output(outputPoints, outputColors, "3DImage.ply")


if __name__ == "__main__":
    reconstruct((1280, 720))
