import os

import numpy as np
import cv2 as cv

from src.fundamental_matrix import fondamentale_huit_points, matrice_fondamentale
from src.rectification import points_opencv, rectification, drawlines
from src.disparity import disparity_map

OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img1 = cv.imread("data/left.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("data/right.jpg", cv.IMREAD_GRAYSCALE)
    pts1, pts2 = points_opencv(img1, img2)

    nb_indices = 50
    indices = np.arange(pts1.shape[0])
    np.random.shuffle(indices)
    indices = indices[:nb_indices]
    pts1_ech = pts1[indices]
    pts2_ech = pts2[indices]

    # Fondamentale à 8 points (sans RANSAC)
    fondamentale_8 = fondamentale_huit_points(pts1[:8], pts2[:8])
    lines0 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fondamentale_8)
    lines0 = lines0.reshape(-1, 3)
    img_8_1, img_8_2 = drawlines(img1, img2, lines0, pts1_ech, pts2_ech)
    cv.imwrite(f"{OUTPUT_DIR}/epilignes_8points_g.png", img_8_1)
    cv.imwrite(f"{OUTPUT_DIR}/epilignes_8points_d.png", img_8_2)

    # Fondamentale d'OpenCV (référence)
    F_cv, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    lines2 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_cv)
    lines2 = lines2.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines2, pts1_ech, pts2_ech)
    cv.imwrite(f"{OUTPUT_DIR}/epilignes_g_opencv.png", img5)
    cv.imwrite(f"{OUTPUT_DIR}/epilignes_d_opencv.png", img6)

    # Fondamentale calculée avec RANSAC maison
    fondamentale_exp, pts1, pts2 = matrice_fondamentale(pts1, pts2)
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fondamentale_exp)
    lines1 = lines1.reshape(-1, 3)
    img3, img4 = drawlines(img1, img2, lines1, pts1, pts2)
    cv.imwrite(f"{OUTPUT_DIR}/epilignes_g.png", img3)
    cv.imwrite(f"{OUTPUT_DIR}/epilignes_d.png", img4)

    # Rectification (avec F OpenCV et avec F calculée à la main)
    _, _, img3_rectified, img4_rectified = rectification(img1, img2, pts1, pts2, F_cv)
    _, _, img1_rectified, img2_rectified = rectification(img1, img2, pts1, pts2, fondamentale_exp)
    cv.imwrite(f"{OUTPUT_DIR}/rectified_1_opencv.png", img3_rectified)
    cv.imwrite(f"{OUTPUT_DIR}/rectified_2_opencv.png", img4_rectified)
    cv.imwrite(f"{OUTPUT_DIR}/rectified_1.png", img1_rectified)
    cv.imwrite(f"{OUTPUT_DIR}/rectified_2.png", img2_rectified)

    # Vérification de la rectification : les lignes épipolaires doivent devenir horizontales
    pts1, pts2 = points_opencv(img1_rectified, img2_rectified)
    indices = np.arange(pts1.shape[0])
    np.random.shuffle(indices)
    indices = indices[:nb_indices]
    pts1 = pts1[indices]
    pts2 = pts2[indices]
    fondamentale_exp, pts1, pts2 = matrice_fondamentale(pts1, pts2)
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fondamentale_exp)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1_rectified, img2_rectified, lines1, pts1, pts2)
    cv.imwrite(f"{OUTPUT_DIR}/verification_rectification_g.png", img5)
    cv.imwrite(f"{OUTPUT_DIR}/verification_rectification_d.png", img6)

    # Carte de disparité
    image_1 = cv.imread(f"{OUTPUT_DIR}/rectified_1.png")
    image_2 = cv.imread(f"{OUTPUT_DIR}/rectified_2.png")
    gray_left = cv.cvtColor(image_1, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(image_2, cv.COLOR_BGR2GRAY)

    stereo = cv.StereoBM_create(numDisparities=64, blockSize=25)
    disp = stereo.compute(gray_left, gray_right)
    cv.imwrite(f"{OUTPUT_DIR}/dmap_opencv.png", disp)

    dmap = disparity_map(gray_left, gray_right, 2)
    disparity = np.uint8(255 - (dmap * 255 / np.max(dmap)))
    cv.imwrite(f"{OUTPUT_DIR}/dmap.png", disparity)


if __name__ == '__main__':
    main()
