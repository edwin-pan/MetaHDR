import numpy as np
import skimage.io as io
from pathlib import Path
from tqdm import tqdm
import time

NUM_IMGS = 450 
IMG_HEIGHT = 1024
IMG_WIDTH = 1024

def get_data(crop=True, crop_factor=0.5):
    # Initialize output
    imgs = None
    if crop:
        new_height = int(IMG_HEIGHT * crop_factor)
        new_width = int(IMG_HEIGHT * crop_factor)
        imgs = np.zeros((4, NUM_IMGS, new_height, new_width, 3))
    else:
        imgs = np.zeros((4, NUM_IMGS, IMG_HEIGHT, IMG_WIDTH, 3))
        
    # Define directories
    hdr_dir = Path(__file__).parent/'data/LDR-HDR-pair_Dataset/HDR'
    ldr_p2_dir = Path(__file__).parent/'data/LDR-HDR-pair_Dataset/LDR_exposure_+2'
    ldr_00_dir = Path(__file__).parent/'data/LDR-HDR-pair_Dataset/LDR_exposure_0'
    ldr_n2_dir = Path(__file__).parent/'data/LDR-HDR-pair_Dataset/LDR_exposure_-2'
    
    print("READING IMGS")
    time.sleep(1)
    
    # Read files
    for i in tqdm(range(NUM_IMGS)):
        index = i + 1
        
        hdr_img = io.imread(hdr_dir/f'HDR_{index:03d}.hdr').astype(np.float64)/255
        ldr_p2_img = io.imread(ldr_p2_dir/f'LDR_{index:03d}.jpg').astype(np.float64)/255
        ldr_00_img = io.imread(ldr_00_dir/f'LDR_{index:03d}.jpg').astype(np.float64)/255
        ldr_n2_img = io.imread(ldr_n2_dir/f'LDR_{index:03d}.jpg').astype(np.float64)/255
        
        if crop:
            # Define crop params
            new_height = int(IMG_HEIGHT * crop_factor)
            new_width = int(IMG_HEIGHT * crop_factor)
            center_height = IMG_HEIGHT // 2
            center_width = IMG_WIDTH // 2
            indeces_height = np.arange(center_height-(new_height//2), center_height+(new_height//2))
            indeces_width = np.arange(center_width-(new_width//2), center_width+(new_width//2))
            
            imgs[0, i, ...] = hdr_img[indeces_height, indeces_width, :]
            imgs[1, i, ...] = ldr_p2_img[indeces_height, indeces_width, :]
            imgs[2, i, ...] = ldr_00_img[indeces_height, indeces_width, :]
            imgs[3, i, ...] = ldr_n2_img[indeces_height, indeces_width, :]
        else:
            imgs[0, i, ...] = hdr_img
            imgs[1, i, ...] = ldr_p2_img
            imgs[2, i, ...] = ldr_00_img
            imgs[3, i, ...] = ldr_n2_img

    
    return imgs

    
    
