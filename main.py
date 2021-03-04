import numpy as np
import matplotlib.pyplot as plt

from src.dataset.read_data import get_data
from src.dataset.dataloader import DataGenerator

batch_size = 5

imgs = get_data()
dl = DataGenerator(imgs)
train, test = dl.sample_batch('meta_train', batch_size)

# May want to delete imgs to save some space...
#del imgs

tr_imgs = train[0]
tr_hdrs = train[1]

ts_imgs = test[0]
ts_hdrs = test[1]

# # Plot a few random ones to ensure (not gamma corrected)
fig, ax = plt.subplots(nrows=batch_size,ncols=4,figsize=(11, 17))
for index in range(batch_size):
    ax[index, 0].imshow(tr_imgs[index*2])
    ax[index, 0].axis('off')
    ax[index, 0].set_title(f'Train Image {index*2}')
    ax[index, 1].imshow(tr_hdrs[index*2])
    ax[index, 1].axis('off')
    ax[index, 1].set_title(f'HDR Image {index*2}')
    
    ax[index, 2].imshow(tr_imgs[index*2 + 1])
    ax[index, 2].axis('off')
    ax[index, 2].set_title(f'Train Image {index*2+1}')
    ax[index, 3].imshow(tr_hdrs[index*2 + 1])
    ax[index, 3].axis('off')
    ax[index, 3].set_title(f'HDR Image {index*2+1}')
    
plt.show()


