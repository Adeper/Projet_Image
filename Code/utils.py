import numpy as np
import skimage as ski

# Bruiter une image
def noise_image(image, noise_type='gaussian'):
    match noise_type:
        case 'gaussian':
            image_noised = ski.util.random_noise(image, mode=noise_type, mean=0, var=0.01) * 255
        case 's&p':
            image_noised = ski.util.random_noise(image, mode=noise_type, amount=0.05) * 255
        case _:
            image_noised = image
    image_noised = image_noised.astype(np.uint8)
    return image_noised
    
# Bruiter une image
def noise_image_pil(image, noise_type='gaussian'):
    # Convertir l'image PIL en tableau numpy
    image_array = np.array(image)

    # Appliquer le bruit
    match noise_type:
        case 'gaussian':
            image_noised = ski.util.random_noise(image_array, mode='gaussian', mean=0, var=0.01) * 255
        case 's&p':
            image_noised = ski.util.random_noise(image_array, mode='s&p', amount=0.05) * 255
        case _:
            image_noised = image_array

    # Convertir en entier 8 bits
    image_noised = image_noised.astype(np.uint8)

    # Convertir le tableau numpy en image PIL
    return Image.fromarray(image_noised)