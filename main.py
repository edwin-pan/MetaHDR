import matplotlib.pyplot as plt

from src.dataset.dataloader import DataGenerator
from src.dataset.hdr_visualization import visualize_hdr_image
from skimage.metrics import structural_similarity as ssim

batch_size = 5

dl = DataGenerator(crop_factor=0.5)
train, test = dl.sample_batch('meta_train', batch_size)

# Each have shape (Batch Size, 2, 512, 512, 3)
tr_imgs = train[0]
tr_hdrs = train[1]

# Each have shape (Batch Size, 1, 512, 512, 3)
ts_imgs = test[0]
ts_hdrs = test[1]

# # Plot 
fig, ax = plt.subplots(nrows=batch_size,ncols=4,figsize=(11, 17))
for index in range(batch_size):
    ax[index, 0].imshow(tr_imgs[index][0])
    ax[index, 0].axis('off')
    ax[index, 0].set_title(f'Train Image {index*2}')
    ax[index, 1].imshow(visualize_hdr_image(tr_hdrs[index][0]))
    ax[index, 1].axis('off')
    ax[index, 1].set_title(f'HDR Image {index*2} (GCed)')
    
    ax[index, 2].imshow(tr_imgs[index][1])
    ax[index, 2].axis('off')
    ax[index, 2].set_title(f'Train Image {index*2+1}')
    ax[index, 3].imshow(visualize_hdr_image(tr_hdrs[index][1]))
    ax[index, 3].axis('off')
    ax[index, 3].set_title(f'HDR Image {index*2+1} (GCed)')
    
plt.show()

total_avg = 0
for index in range(batch_size):
    ssim_1 = round(ssim(tr_hdrs[index][0], tr_imgs[index][0], multichannel=True),4)
    ssim_2 = round(ssim(tr_hdrs[index][1], tr_imgs[index][1], multichannel=True), 4)
    ssim_3 = round(ssim(ts_hdrs[index][0], ts_imgs[index][0], multichannel=True), 4)
    avg = round((ssim_1 + ssim_2 + ssim_3) / 3, 4)
    total_avg += avg
    print(f"SSIM FOR INDEX {index} THREE EXPOSURES: {ssim_1}, {ssim_2}, & {ssim_3} AND AVERAGE: {avg}")
total_avg /= batch_size
print("TOTAL AVG SSIM:", round(total_avg, 4))
