import numpy as np
import skimage.io as io
from pathlib import Path
from tqdm import tqdm
import time

NUM_IMGS = 450 
IMG_HEIGHT = 1024
IMG_WIDTH = 1024

def normalize_hdr(img):
    return (img - img.min()) / (img.max() - img.min())


def get_data(crop=True, crop_factor=0.5):
    """
    Reads the data into numpy arrays
    Args:
     crop: a boolean which says if we need to crop the input images
     crop_factor: how much to crop by (will only be needed if crop is True)
    """
    
    
    # Initialize output
    imgs = None
    if crop:
        new_height = int(IMG_HEIGHT * crop_factor)
        new_width = int(IMG_HEIGHT * crop_factor)
        imgs = np.zeros((NUM_IMGS, 4, new_height, new_width, 3))
    else:
        imgs = np.zeros((NUM_IMGS, 4, IMG_HEIGHT, IMG_WIDTH, 3))
        
    # Define directories
    hdr_dir = Path(__file__).parent.parent.parent/'data/LDR-HDR-pair_Dataset/HDR'
    ldr_p2_dir = Path(__file__).parent.parent.parent/'data/LDR-HDR-pair_Dataset/LDR_exposure_+2'
    ldr_00_dir = Path(__file__).parent.parent.parent/'data/LDR-HDR-pair_Dataset/LDR_exposure_0'
    ldr_n2_dir = Path(__file__).parent.parent.parent/'data/LDR-HDR-pair_Dataset/LDR_exposure_-2'
    
    print("READING IMGS...")
    time.sleep(1)
    
    # Read files
    for i in tqdm(range(NUM_IMGS)):
        index = i + 1
        
        hdr_img = io.imread(hdr_dir/f'HDR_{index:03d}.hdr').astype(np.float64)
        ldr_p2_img = io.imread(ldr_p2_dir/f'LDR_{index:03d}.jpg').astype(np.float64)/255
        ldr_00_img = io.imread(ldr_00_dir/f'LDR_{index:03d}.jpg').astype(np.float64)/255
        ldr_n2_img = io.imread(ldr_n2_dir/f'LDR_{index:03d}.jpg').astype(np.float64)/255
        if crop:
            # Define crop params
            new_height = int(IMG_HEIGHT * crop_factor)
            new_width = int(IMG_HEIGHT * crop_factor)
            center_height = IMG_HEIGHT // 2
            center_width = IMG_WIDTH // 2
            
            h1, h2 = center_height-(new_height//2), center_height+(new_height//2)
            w1, w2 = center_width-(new_width//2), center_width+(new_width//2)

            imgs[i, 0, ...] = hdr_img[h1:h2, w1:w2, :]            
            imgs[i, 1, ...] = ldr_p2_img[h1:h2, w1:w2, :]
            imgs[i, 2, ...] = ldr_00_img[h1:h2, w1:w2, :]
            imgs[i, 3, ...] = ldr_n2_img[h1:h2, w1:w2, :]
        else:
            imgs[i, 0, ...] = hdr_img
            imgs[i, 1, ...] = ldr_p2_img
            imgs[i, 2, ...] = ldr_00_img
            imgs[i, 3, ...] = ldr_n2_img

    
    return imgs

    
    
