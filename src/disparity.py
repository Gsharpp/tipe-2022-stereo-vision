import numpy as np
from threading import Thread, Lock


def remplit(img, hauteur, largeur):
    """
    Obj : remplit une image (en bas à droite) avec des zéros
    Entrée : une image, la hauteur finale, la largeur finale"
    Sortie : la nouvelle image
    """
    h, l = np.shape(img)
    assert (h <= hauteur and l <= largeur)
    new_image = np.zeros((hauteur, largeur))

    for i in range(hauteur):
        for j in range(largeur):
            if i < h:
                new_image[i, j] = img[i, j]
            else:
                new_image[i, j] = 0

    return new_image


def ajuste(img1, img2, k):
    """
    Obj : ajuster les images en ajoutant des bordures pour faciliter
            la recherche des pixels (on ajoute autant en
            hauteur qu'en largeur)
    Entrée : les deux images, de gauche et de droite et l'entier k,
            la taille de la zone de recherche depuis le pixel
    Sortie : les deux images modifiées, et leur dimension
    """
    h1, l1 = np.shape(img1)
    h2, l2 = np.shape(img2)
    hauteur = max(h1, h2)
    largeur = max(l1, l2)
    if h1 != h2 or l1 != l2:
        img1 = remplit(img1, hauteur, largeur)

    img_bordure_1 = np.zeros((hauteur + 2*k, largeur + 2*k))
    img_bordure_2 = np.zeros((hauteur + 2*k, largeur + 2*k))

    for i in range(hauteur):
        for j in range(largeur):
            img_bordure_1[i+k, j+k] = img1[i, j]
            img_bordure_2[i+k, j+k] = img2[i, j]
    return img_bordure_1, img_bordure_2, (hauteur, largeur)


def recherche_droite(img_gauche, img_droite, pixel_gauche, k):
    """
    Obj : Rechercher un pixel de l'image de gauche dans l'image de droite
    Entrée : Les 2 images, les coordonnées du pixel de gauche, l'entier k paramètre
    de la taille du cadre de recherche
    """
    meilleur_match = 0
    valeur_match = float("inf")
    zone_gauche = img_gauche[pixel_gauche[0] - k: pixel_gauche[0] + k+1,
                              pixel_gauche[1] - k: pixel_gauche[1] + k+1]

    for colonne in range(k, pixel_gauche[1]):
        zone_droite = img_droite[pixel_gauche[0] - k: pixel_gauche[0] + k+1,
                                  colonne - k: colonne + k+1]

        # comparaison_zones = np.sum((zone_gauche - zone_droite) ** 2) # SSD
        comparaison_zones = np.sum(abs(zone_gauche - zone_droite))  # SAD
        if comparaison_zones < valeur_match:
            meilleur_match = colonne
            valeur_match = comparaison_zones

    return pixel_gauche[1] - meilleur_match


def disparity_map(img_gauche, img_droite, k):
    """
    Obj : calculer la disparity map (carte de disparité) de
            deux images rectifiées par bloc matching
    Entrée : les deux images, un entier k la distance
            avec laquelle on cherche autour de chaque pixel
    Sortie : la disparity map
    """
    img_gauche, img_droite, dim = ajuste(img_gauche, img_droite, k)
    hauteur, largeur = dim
    dmap = np.zeros((hauteur, largeur))
    compteur = 0

    for ligne in range(k, hauteur):
        for colonne in range(k, largeur):
            pixel = (ligne, colonne)
            dist = recherche_droite(img_gauche, img_droite, pixel, k)
            dmap[ligne, colonne] = dist

        compteur += 1
        print(str(compteur) + " / " + str(hauteur))

    return dmap


def disparity_map_multithreading(img_gauche, img_droite, k, nb_threads=8):
    """
    Obj : calculer la disparity map (carte de disparité) de deux images rectifiées
            par bloc matching, en répartissant les lignes sur plusieurs threads
    Entrée : les deux images, un entier k la distance avec laquelle on cherche autour
            de chaque pixel, le nombre de threads à utiliser
    Sortie : la disparity map
    """
    img_gauche, img_droite, dim = ajuste(img_gauche, img_droite, k)
    hauteur, largeur = dim
    dmap = np.zeros((hauteur, largeur))
    compteur = 0
    mutex = Lock()
    threads = []

    def traitement_ligne(debut_ligne, fin_ligne):
        nonlocal compteur
        for ligne in range(debut_ligne, fin_ligne):
            for colonne in range(k, largeur):
                pixel = (ligne, colonne)
                dist = recherche_droite(img_gauche, img_droite, pixel, k)
                dmap[ligne, colonne] = dist

            mutex.acquire()
            compteur += 1
            print(str(compteur) + " / " + str(hauteur))
            mutex.release()

    q = hauteur // nb_threads
    r = hauteur % nb_threads

    for i in range(nb_threads):
        t = Thread(target=traitement_ligne, args=(k + i*q, k + (i+1)*q))
        threads.append(t)
        t.start()

    if r > 0:
        t = Thread(target=traitement_ligne, args=(k + nb_threads*q, hauteur))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return dmap
