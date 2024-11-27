import numpy as np
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from scipy.signal import wiener
import pywt

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

# Filtre moyenneur
def mean_denoise(image_noised, window_size=3):
    image_noised = np.array(image_noised)
    image_noised = image_noised / 255
    if image_noised.ndim not in [2, 3]:
        raise ValueError("L'image d'entrée doit être en noir et blanc (1 canal) ou en couleur (3 canaux).")
    if window_size % 2 == 0:
        raise ValueError("La taille de la fenêtre doit être impaire.")

    image_filtree = image_noised.copy()
    offset = window_size // 2

    if image_noised.ndim == 2:
        for i in range(offset, image_noised.shape[0] - offset):
            for j in range(offset, image_noised.shape[1] - offset):
                window = image_noised[i - offset:i + offset + 1, j - offset:j + offset + 1]
                median_value = np.mean(window)
                image_filtree[i, j] = median_value

    elif image_noised.ndim == 3 and image_noised.shape[2] == 3:
        for channel in range(3):
            for i in range(offset, image_noised.shape[0] - offset):
                for j in range(offset, image_noised.shape[1] - offset):
                    window = image_noised[i - offset:i + offset + 1, j - offset:j + offset + 1, channel]
                    median_value = np.median(window)
                    image_filtree[i, j, channel] = median_value
    else:
        raise ValueError("Format d'image non supporté.")

    return image_filtree

# Filtre bilatéral
def bilateral_denoise(image_noised, sigma_color=0.1, sigma_spatial=15):
    image_noised_np = np.array(image_noised, dtype=np.float64)
    
    image_noised_np /= 255.0
    
    if image_noised_np.ndim == 2: 
        image_denoised_np = denoise_bilateral(image_noised_np, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    elif image_noised_np.ndim == 3:  
        image_denoised_np = np.zeros_like(image_noised_np)
        for channel in range(3):
            image_denoised_np[:, :, channel] = denoise_bilateral(image_noised_np[:, :, channel], sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    else:
        raise ValueError("Format d'image non supporté.")
    
    return image_denoised_np

# Filtre de Wiener
def wiener_denoise(image_noised, kernel_size=3):
    image_noised_np = np.array(image_noised, dtype=np.float64)
    image_noised_np /= 255.0
    
    if image_noised_np.ndim == 2:
        image_denoised_np = wiener(image_noised_np, (kernel_size, kernel_size))
    elif image_noised_np.ndim == 3:
        image_denoised_np = np.zeros_like(image_noised_np)
        for channel in range(3):
            image_denoised_np[:, :, channel] = wiener(image_noised_np[:, :, channel], (kernel_size, kernel_size))
    else:
        raise ValueError("Format d'image non supporté.")
    
    image_denoised_np = np.clip(image_denoised_np, 0, 1)
    
    return image_denoised_np

# Variation totale
def total_variation_denoise(image_noised, weight=0.1):
    image_noised_np = np.array(image_noised, dtype=np.float64)
    image_noised_np /= 255.0
    
    if image_noised_np.ndim == 2:
        image_denoised_np = denoise_tv_chambolle(image_noised_np, weight=weight)
    elif image_noised_np.ndim == 3:
        image_denoised_np = np.zeros_like(image_noised_np)
        for channel in range(3):
            image_denoised_np[:, :, channel] = denoise_tv_chambolle(image_noised_np[:, :, channel], weight=weight)
    else:
        raise ValueError("Format d'image non supporté.")
    
    return image_denoised_np