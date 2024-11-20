import numpy as np

# Filtre médian
def median_denoise(image_noised, window_size=3):
    # Vérification de la validité de l'image et de la taille de la fenêtre
    image_noised = np.array(image_noised)
    image_noised = image_noised / 255
    if image_noised.ndim not in [2, 3]:
        raise ValueError("L'image d'entrée doit être en noir et blanc (1 canal) ou en couleur (3 canaux).")
    if window_size % 2 == 0:
        raise ValueError("La taille de la fenêtre doit être impaire.")

    # Initialisation
    image_filtree = image_noised.copy()
    offset = window_size // 2

    # Si l'image est en noir et blanc (1 canal)
    if image_noised.ndim == 2:
        for i in range(offset, image_noised.shape[0] - offset):
            for j in range(offset, image_noised.shape[1] - offset):
                window = image_noised[i - offset:i + offset + 1, j - offset:j + offset + 1]
                median_value = np.median(window)
                image_filtree[i, j] = median_value

    # Si l'image est en couleur (3 canaux)
    elif image_noised.ndim == 3 and image_noised.shape[2] == 3:
        for channel in range(3):  # Boucle sur les canaux R, G, B
            for i in range(offset, image_noised.shape[0] - offset):
                for j in range(offset, image_noised.shape[1] - offset):
                    window = image_noised[i - offset:i + offset + 1, j - offset:j + offset + 1, channel]
                    median_value = np.median(window)
                    image_filtree[i, j, channel] = median_value
    else:
        raise ValueError("Format d'image non supporté.")

    return image_filtree
