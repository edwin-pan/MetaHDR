import numpy as np
import skimage.io as io
from pathlib import Path
from tqdm import tqdm
import time
from scipy.interpolate import interp1d

NUM_IMGS = 450 
IMG_HEIGHT = 1024
IMG_WIDTH = 1024

def get_darkdetail_data(hdr_image, hdr_max_val):
    """
    Assuming HDR capped at 4.0
    
    """
    
    global_linear_mapping = interp1d([0,hdr_max_val],[0,1])
    
    # Ceiling values > hdr_max_val
    copy_img = hdr_image.copy()
    copy_img[copy_img>hdr_max_val] = hdr_max_val
    
    return global_linear_mapping(copy_img)


def get_brightdetail_data(hdr_image, hdr_min_val):
    """
    Assuming HDR capped at 4.0
    
    """
    
    global_linear_mapping = interp1d([hdr_min_val,4.0],[0,1])
    
    # Floor values < hdr_min_val
    copy_img = hdr_image.copy()
    copy_img[copy_img<hdr_min_val] = hdr_min_val
    
    return global_linear_mapping(copy_img)


def get_middetail_data(hdr_image, hdr_min_val, hdr_max_val):
    """
    Assuming HDR capped at 4.0
    
    """
    
    global_linear_mapping = interp1d([hdr_min_val,hdr_max_val],[0,1])
    
    # Clipping
    copy_img = hdr_image.copy()
    copy_img[copy_img<hdr_min_val] = hdr_min_val
    copy_img[copy_img>hdr_max_val] = hdr_max_val
    
    return global_linear_mapping(copy_img)


def get_data(crop=True, crop_factor=0.5, num_exposures=3):
    """
    Reads the data into a numpy array. Could produce mixture of simulated and real exposures.
    Args:
     crop: a boolean which says if we need to crop the input images
     crop_factor: how much to crop by (will only be needed if crop is True)
     num_exposures: If more than 3, will add simulated data.
    """
    
    assert num_exposures > 2, "Should have at least 3 exposures"
    
    sim_min = 0.75 # -> 4.0
    sim_max = 0.5
    
    # Initialize output
    imgs = None
    if crop:
        new_height = int(IMG_HEIGHT * crop_factor)
        new_width = int(IMG_HEIGHT * crop_factor)
        imgs = np.zeros((NUM_IMGS, num_exposures+1, new_height, new_width, 3))
    else:
        imgs = np.zeros((NUM_IMGS, num_exposures+1, IMG_HEIGHT, IMG_WIDTH, 3))
    
    
    # 256*16*64*2*2*4*128
    
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

            if num_exposures > 3:
                for j in range(4,num_exposures+1):
                    if j%2:
                        imgs[i,j, ...] = get_brightdetail_data(hdr_img, hdr_min_val=sim_min)[h1:h2, w1:w2, :]
                    else:
                        imgs[i,j, ...] = get_darkdetail_data(hdr_img, hdr_max_val=sim_max)[h1:h2, w1:w2, :]
                        
        else:
            imgs[i, 0, ...] = hdr_img
            imgs[i, 1, ...] = ldr_p2_img
            imgs[i, 2, ...] = ldr_00_img
            imgs[i, 3, ...] = ldr_n2_img
            
            if num_exposures > 3:
                for j in range(4,num_exposures+1):
                    if j%2:
                        imgs[i,j, ...] = get_brightdetail_data(hdr_img, hdr_min_val=sim_min)
                    else:
                        imgs[i,j, ...] = get_darkdetail_data(hdr_img, hdr_max_val=sim_max)
    
    return imgs

    
    
