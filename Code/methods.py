import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import bm3d
from scipy.signal import wiener
import pywt
import torch
import torch.nn as nn
from pathlib import Path
import os
import sys
from os import path as osp
import numpy as np
from basicsr.models import build_model
from basicsr.utils import tensor2img
import yaml
from basicsr.utils.options import ordered_yaml
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
import logging
from basicsr.utils.options import dict2str

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

    image_denoised_np = np.clip(image_denoised_np, 0, 1)
    
    return image_denoised_np

# Ondelettes de Haar
def haar_denoise(image_noised, threshold=0.1):
    image_noised_np = np.array(image_noised, dtype=np.float64)
    image_noised_np = image_noised_np / 255.0
    
    if image_noised_np.ndim == 3 and image_noised_np.shape[2] == 3:
        image_denoised = np.zeros_like(image_noised_np)
        
        for canal in range(3):
            channel = image_noised_np[:, :, canal]
            
            coeffs2 = pywt.dwt2(channel, 'haar')
            LL, (LH, HL, HH) = coeffs2
            
            LH = pywt.threshold(LH, threshold, mode='soft')
            HL = pywt.threshold(HL, threshold, mode='soft')
            HH = pywt.threshold(HH, threshold, mode='soft')
            
            denoised_channel = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
            
            image_denoised[:, :, canal] = np.clip(denoised_channel, 0, 1)  # Normaliser entre 0 et 1
        
        #image_denoised = np.clip(image_denoised, 0, 1)
        image_denoised = np.uint8(image_denoised * 255)
        
        return image_denoised

    else:
        raise ValueError("L'image doit être en couleur (3 canaux).")

# Filtre BM3D
def bm3d_denoise(image_noised, sigma_psd=25, stage_arg=0.1):

    image_noised_np = np.array(image_noised, dtype=np.float64)
    image_noised_np = image_noised_np / 255.0
    
    if image_noised_np.ndim == 2:
        image_denoised_np = bm3d.bm3d(image_noised_np, sigma_psd=ecart_type_bruit)
    elif image_noised_np.ndim == 3:
        image_denoised_np = np.zeros_like(image_noised_np)
        for canal in range(3):
            image_denoised_np[:, :, canal] = bm3d.bm3d(image_noised_np[:, :, canal], sigma_psd=sigma_psd)
    else:
        raise ValueError("Format d'image non supporté.")
    
    return image_denoised_np

class CGNet(nn.Module):
    def __init__(self):
        super(CGNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)

class GANDenoise:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        return model

    def denoise(self, image_noised):
        input_tensor = self._preprocess_image(image_noised).to(self.device)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        denoised_image = self._postprocess_image(output_tensor)
        return denoised_image

    def _preprocess_image(self, image):
        image = np.array(image, dtype=np.float32) / 255.0
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        else:
            image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image, dtype=torch.float32)

    def _postprocess_image(self, output_tensor):
        output_image = output_tensor.squeeze().cpu().numpy()
        if output_image.ndim == 3:
            output_image = np.transpose(output_image, (1, 2, 0))
        return np.clip(output_image, 0, 1)

# def yaml_load(f):
#     with open(f, mode='r') as file:
#         Loader, Dumper = ordered_yaml()
#         return yaml.load(file, Loader=Loader)

# class CGNetDenoise:
#     def __init__(self, yml_path, model_path):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self._load_model(yml_path, model_path)

#     def _load_model(self, yml_path, model_path):
#         opt_path = Path(yml_path)
#         opt = yaml_load(opt_path)
#         opt['dist'] = False
#         opt['is_train'] = False
#         opt['path']['pretrain_network_g'] = model_path

#         model = build_model(opt)
#         model.net_g.to(self.device)
#         model.net_g.eval()
#         return model

#     def denoise(self, image_noised):
#         input_tensor = self._preprocess_image(image_noised).to(self.device)

#         with torch.no_grad():
#             output_tensor = self.model.net_g(input_tensor.unsqueeze(0))

#         return self._postprocess_image(output_tensor.squeeze(0))

#     def _preprocess_image(self, image):
#         image = np.array(image, dtype=np.float32) / 255.0
#         if image.ndim == 2:
#             image = np.expand_dims(image, axis=0)
#         else:
#             image = np.transpose(image, (2, 0, 1))
#         return torch.tensor(image, dtype=torch.float32)

#     def _postprocess_image(self, output_tensor):
#         output_image = output_tensor.cpu().numpy()
#         if output_image.ndim == 3:
#             output_image = np.transpose(output_image, (1, 2, 0))
#         return np.clip(output_image, 0, 1)