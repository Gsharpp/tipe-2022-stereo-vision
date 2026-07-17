import numpy as np


def fondamentale_huit_points(pts1, pts2):
    """
    Obj : Calcule la matrice fondamentale pour 8 points donnés (algorithme à huit points)
    Entrée : Les coordonnées de 8 points correspondants deux à deux
    Sortie : Une estimation de la matrice fondamentale
    """
    # Réarrange les coordonnées dans une matrice
    A = np.zeros((len(pts1), 9))
    for i in range(8):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]
        A[i] = [x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    _, _, V = np.linalg.svd(A)
    F = V[-1, :]
    F = F.reshape(3, 3)
    Uf, D, Vf = np.linalg.svd(F)
    D = np.diag(D)
    D[2, 2] = 0
    F = np.dot(Uf, np.dot(D, Vf))
    return F


def calcul_erreur(pts1, pts2, F):
    """
    Calcule l'erreur de la matrice pour deux points
    """
    u = np.asarray([pts1[0], pts1[1], 1])
    v = np.asarray([pts2[0], pts2[1], 1])
    erreur = np.dot(v.T, np.dot(F, u))
    erreur = abs(erreur)
    return erreur


def matrice_fondamentale(pt1, pt2):
    """
    Obj : Calculer la matrice fondamentale du système avec l'algorithme à huit points
    Entrée : 2 tableaux de points se correspondants sur les deux images
    Sortie : La matrice fondamentale du système, les meilleurs points associés
    """
    indices = np.arange(pt1.shape[0])
    F = np.zeros
    nb_points = pt1.shape[0]
    coherents = 0
    best_pts1 = []
    best_pts2 = []
    for i in range(2000):
        indices_t = []
        np.random.shuffle(indices)  # On prend 8 points au hasard
        indices_8 = indices[:8]
        new_F = fondamentale_huit_points(pt1[indices_8], pt2[indices_8])  # Estimation de F

        for j in range(nb_points):
            erreur = calcul_erreur(pt1[j], pt2[j], new_F)
            if erreur < 0.001:
                indices_t.append(j)

        if len(indices_t) > coherents:
            coherents = len(indices_t)
            F = new_F
            best_pts1 = pt1[indices_t]
            best_pts2 = pt2[indices_t]
    return F, best_pts1, best_pts2
