# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import argparse
from os import path as osp

from basicsr.utils import scandir_SIDD
from basicsr.utils.create_lmdb import create_lmdb_for_SIDD
from basicsr.utils.lmdb_util import make_lmdb_from_imgs
from basicsr.utils.create_lmdb import prepare_keys, make_lmdb_from_imgs

def create_custom_lmdb(folder_path, lmdb_path):
    """
    Create LMDB for a given folder of images.
    
    Args:
        folder_path (str): Path to the folder containing images.
        lmdb_path (str): Path to save the LMDB file.
    """
    if not osp.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist.")
    
    img_path_list, keys = prepare_keys(folder_path, 'PNG')  # Assumes PNG format
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def main():
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    opt['input_folder'] = './datasets/SIDD/Data2'
    opt['crop_size'] = 512
    opt['step'] = 384
    opt['thresh_size'] = 0

    # Traitement des images NOISY
    opt['save_folder'] = './datasets/SIDD/val/input_crops'
    opt['keywords'] = 'NOISY_SRGB'  # Ajusté pour votre structure
    #extract_subimages(opt)

    # Traitement des images GT
    opt['save_folder'] = './datasets/SIDD/val/gt_crops'
    opt['keywords'] = 'GT_SRGB'  # Ajusté pour votre structure
    #extract_subimages(opt)

    # Création des LMDB
    create_lmdb_for_SIDD()
    create_custom_lmdb('./datasets/SIDD/val/input_crops', './datasets/SIDD/val/input_crops.lmdb')
    create_custom_lmdb('./datasets/SIDD/val/gt_crops', './datasets/SIDD/val/gt_crops.lmdb')



def extract_subimages(opt):
    """Crop images to subimages."""
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Continuing.')

    # Recherche récursive dans tous les sous-dossiers
    img_list = [
        osp.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files if opt['keywords'] in file
    ]

    print(f"Found {len(img_list)} images matching '{opt['keywords']}' in {input_folder}")

    if not img_list:
        print(f"No images found with keywords '{opt['keywords']}' in {input_folder}.")
        sys.exit(1)

    # Découpage des images en parallèle
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda _: pbar.update(1)
        )
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.
    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.
    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    img_name = img_name.replace(opt['keywords'], '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
    # ... make sidd to lmdb