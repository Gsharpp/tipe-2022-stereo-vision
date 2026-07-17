"""
Outil optionnel de calibrage manuel : sélection à la souris de points
correspondants entre deux images juxtaposées, en alternative à la détection
automatique SIFT/FLANN utilisée par le pipeline principal (main.py).
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

from src.fundamental_matrix import fondamentale_huit_points


def juxtaposition(im_g, im_d):
    """
    Obj : Juxtapose deux images de même hauteur
    Entrée : deux images de mêmes dimensions
    Sortie : La concaténation des deux images
    """
    hg, lg = np.shape(im_g)
    hd, ld = np.shape(im_d)

    assert hg == hd, "Problème de hauteur des images"

    new_image = np.zeros((hg, lg + ld, 3), np.uint8)

    for i in range(hg - 1):
        for j in range(lg - 1):
            new_image[i, j] = im_g[i, j]  # Partie gauche

    for i in range(hg - 1):
        for j in range(ld - 1):
            new_image[i, j+lg] = im_d[i, j]  # Partie droite

    return new_image


def recuperation_points(img_g, img_d):
    """
    Obj : Récupère manuellement des points correspondant entre deux images juxtaposées
    Entrée : Deux images de même hauteur
    Sortie : 2 listes de coordonnées, d'au moins 8 points corrélés entre eux (clic gauche pour
        pointer, 'q' pour valider)
    """
    l_img_g, _ = np.shape(img_g)

    image_concat = juxtaposition(img_g, img_d)

    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(np.array([x, y, 1]))

            # Afficher les coordonnées sur l'image en texte
            cv2.putText(image_concat, f'({x},{y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Dessiner un rond sur l'image autour de la zone cliquée
            cv2.circle(image_concat, (x, y), 3, (0, 255, 255), -1)

    cv2.namedWindow('Point Coordinates')
    cv2.setMouseCallback('Point Coordinates', click_event)

    while True:
        cv2.imshow("Point Coordinates", image_concat)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    assert len(points) % 2 == 0, "Problème sur les points de calibrage"

    for i in range(len(points)//2):
        x, y, _ = points[2*i+1]
        points[2*i+1] = np.array([x, y - l_img_g, 1])

    return points[::2], points[1::2]


def Features(image1, image2):
    """
    Obj : Alternative à la détection SIFT+FLANN : appariement de points d'intérêt par ORB
    Entrée : les deux images
    Sortie : 2 tableaux de points se correspondant deux à deux, triés par qualité du match
    """
    img1_copy = image1.copy()
    img2_copy = image2.copy()

    image1_gray = cv2.cvtColor(img1_copy, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(img2_copy, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)

    kp1, f1 = orb.detectAndCompute(image1_gray, None)
    kp2, f2 = orb.detectAndCompute(image2_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    points = bf.match(f1, f2)

    matches = sorted(points, key=lambda x: x.distance)

    match_img = cv2.drawMatches(img1_copy, kp1, img2_copy, kp2, matches[:1000], None)

    plt.figure(1)
    plt.title('Matches')
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))

    pt1 = np.float32([kp1[i.queryIdx].pt for i in matches])
    pt2 = np.float32([kp2[j.trainIdx].pt for j in matches])

    return pt1, pt2


def matrice_essentielle(F, K1, K2):
    """
    Obj : Calcule la matrice essentielle
    Entrées : La matrice fondamentale, matrice de la caméra 1, matrice de la caméra 2
    Sortie : Matrice essentielle
    """
    E = np.dot(K2.T, np.dot(F, K1))
    return E


if __name__ == "__main__":
    image_1 = cv2.imread("data/left.jpg")
    image_2 = cv2.imread("data/right.jpg")

    gris1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    pts1, pts2 = recuperation_points(gris1, gris2)

    F = fondamentale_huit_points(np.array(pts1), np.array(pts2))
    print("Matrice fondamentale (calibrage manuel) :")
    print(F)
