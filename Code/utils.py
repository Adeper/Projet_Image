import numpy as np
import skimage as ski
from PIL import Image
from skimage.metrics import structural_similarity as ssim

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

# Bruiter une image (version pour l'app)
def noise_image_pil(image, noise_strength, noise_type='gaussian'):
    image_array = np.array(image)

    image_array = image_array / 255.0

    match noise_type:
        case 'gaussian':
            image_noised = ski.util.random_noise(image_array, mode='gaussian', mean=0, var=noise_strength)
        case 's&p':
            image_noised = ski.util.random_noise(image_array, mode='s&p', amount=noise_strength)
        case _:
            image_noised = image_array

    image_noised = (image_noised * 255).astype(np.uint8)

    return Image.fromarray(image_noised)

def psnr(image1, image2):
    if isinstance(image2, np.ndarray):
        image2 = np.copy(image2)
        if image2.dtype != np.uint8:
            image2 = (np.clip(image2, 0, 1) * 255).astype(np.uint8)
        image2 = Image.fromarray(image2)
    image_2_resized = image2.resize(image1.size)
    array1 = np.array(image1)
    array2 = np.array(image_2_resized)
    if array1.shape != array2.shape:
        raise ValueError("Les deux images doivent avoir la même dimension pour calculer le PSNR.")
    return ski.metrics.peak_signal_noise_ratio(array1, array2)

def ssim_score(image1, image2):
    if isinstance(image2, np.ndarray):
        image2 = np.copy(image2)
        if image2.dtype != np.uint8:
            image2 = (np.clip(image2, 0, 1) * 255).astype(np.uint8)
        image2 = Image.fromarray(image2)
    if image1.size != image2.size:
        image_2_resized = image2.resize(image1.size)
    else:
        image_2_resized = image2
    array1 = np.array(image1)
    array2 = np.array(image_2_resized)
    if array1.shape != array2.shape:
        raise ValueError("Les deux images doivent avoir la même dimension pour calculer le SSIM.")
    if array1.max() > 1:
        array1 = array1 / 255.0
        array2 = array2 / 255.0
    data_range = 1.0 if array1.max() <= 1 else 255.0
    ssim_value = ssim(array1, array2, multichannel=True, channel_axis=2, data_range=data_range)
    return ssim_value