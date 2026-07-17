def depthmap(baseline, focale, disparity_map):
    """
    Obj : calculer la depthmap (carte de profondeur), grâce à la carte de disparité
    Entrée : la distance entre les deux caméras, leur focale, la carte de disparité obtenue
        précédemment
    Sortie : la depthmap
    """
    # Calcule pour chaque pixel leur profondeur
    d = (baseline * focale) / (disparity_map)
    return d
