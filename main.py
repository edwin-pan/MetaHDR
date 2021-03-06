import numpy as np
import matplotlib.pyplot as plt

from src.dataset.dataloader import DataGenerator

batch_size = 5

dl = DataGenerator()
train, test = dl.sample_batch('meta_train', batch_size)

# Each have shape (Batch Size, 2, 512, 512, 3)
tr_imgs = train[0]
tr_hdrs = train[1]

# Each have shape (Batch Size, 1, 512, 512, 3)
ts_imgs = test[0]
ts_hdrs = test[1]

# # Plot a few random ones to ensure (not gamma corrected)
fig, ax = plt.subplots(nrows=batch_size,ncols=4,figsize=(11, 17))
for index in range(batch_size):
    ax[index, 0].imshow(tr_imgs[index][0])
    ax[index, 0].axis('off')
    ax[index, 0].set_title(f'Train Image {index*2}')
    ax[index, 1].imshow(tr_hdrs[index][0])
    ax[index, 1].axis('off')
    ax[index, 1].set_title(f'HDR Image {index*2}')
    
    ax[index, 2].imshow(tr_imgs[index][1])
    ax[index, 2].axis('off')
    ax[index, 2].set_title(f'Train Image {index*2+1}')
    ax[index, 3].imshow(tr_hdrs[index][1])
    ax[index, 3].axis('off')
    ax[index, 3].set_title(f'HDR Image {index*2+1}')
    
plt.show()


