import numpy as np
import cv2 as cv


def points_opencv(img1, img2):
    """
    Obj : Détecte et apparie des points d'intérêt entre deux images (SIFT + FLANN)
    Entrée : les deux images
    Sortie : 2 tableaux de points se correspondant deux à deux
    """
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2


def rectification(img1, img2, pts1, pts2, F):
    """
    Obj : Rectifier les deux images pour qu'elles soient translatées (avec OpenCV)
    Entrée : les deux images gauche et droite, deux sets de points se correspondant, la matrice
        fondamentale du système
    Sortie : les sets de points des images rectifiées, les images rectifiées
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))

    rectified_pts1 = np.zeros((pts1.shape), dtype=int)
    rectified_pts2 = np.zeros((pts2.shape), dtype=int)
    for i in range(pts1.shape[0]):
        source1 = np.array([pts1[i][0], pts1[i][1], 1])
        new_point1 = np.dot(H1, source1)
        new_point1[0] = int(new_point1[0]/new_point1[2])
        new_point1[1] = int(new_point1[1]/new_point1[2])
        new_point1 = np.delete(new_point1, 2)
        rectified_pts1[i] = new_point1
        source2 = np.array([pts2[i][0], pts2[i][1], 1])
        new_point2 = np.dot(H2, source2)
        new_point2[0] = int(new_point2[0]/new_point2[2])
        new_point2[1] = int(new_point2[1]/new_point2[2])
        new_point2 = np.delete(new_point2, 2)
        rectified_pts2[i] = new_point2
    img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
    return rectified_pts1, rectified_pts2, img1_rectified, img2_rectified


def drawlines(img1, img2, lines, pts1, pts2):
    """ img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines """
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2
