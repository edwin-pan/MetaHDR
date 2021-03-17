import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset.dataloader import DataGenerator
from src.dataset.hdr_visualization import visualize_hdr_image
from piqa import SSIM, PSNR

batch_size = 5
num_exposures = 3

# dl = DataGenerator(crop_factor=0.5, num_exposures=num_exposures)
# train, test = dl.sample_batch('meta_train', batch_size)

# # Each have shape (Batch Size, num_exposures - 1, 512, 512, 3)
# tr_imgs = train[0]
# tr_hdrs = train[1]

# # Each have shape (Batch Size, 1, 512, 512, 3)
# ts_imgs = test[0]
# ts_hdrs = test[1]

# # # Plot 
# fig, ax = plt.subplots(nrows=batch_size,ncols=2*(num_exposures-1),figsize=(5*(num_exposures-1), 17))
# for index in range(batch_size):
#     for i in range(num_exposures - 1):
#         ax[index, 2*i].imshow(tr_imgs[index][i])
#         ax[index, 2*i].axis('off')
#         ax[index, 2*i].set_title(f'Train Image {index*2 + i}')
#         ax[index, 2*i+1].imshow(visualize_hdr_image(tr_hdrs[index][i]))
#         ax[index, 2*i+1].axis('off')
#         ax[index, 2*i+1].set_title(f'HDR Image {index*2 + i} (GCed)')    
# plt.show()

# total_avg = 0
# for index in range(batch_size):
#     ssims = []
#     for i in range(num_exposures - 1):
#         ssims.append(round(ssim(tr_hdrs[index][i], tr_imgs[index][i], multichannel=True), 4))
#     avg = round(np.mean(ssims), 4)
#     total_avg += avg
#     print(f"TRAIN SSIMS FOR INDEX {index}: {ssims} AND AVERAGE: {avg}")
# total_avg /= batch_size
# print("TOTAL AVG SSIM:", round(total_avg, 4))

dl = DataGenerator(crop_factor=0.5, num_exposures=num_exposures)
data = dl.meta_test_data

ssim = SSIM().double()
psnr = PSNR().double()

total_ssim = 0 
total_psnr = 0

for i in range(data.shape[0]):
    hdr_image = data[i, 0, ...]
    exposure1 = data[i, 1, ...]
    exposure2 = data[i, 2, ...]
    exposure3 = data[i, 3, ...]
    
    exposure1 = torch.from_numpy(exposure1).permute(2, 0, 1).unsqueeze(0)
    exposure2 = torch.from_numpy(exposure2).permute(2, 0, 1).unsqueeze(0)
    exposure3 = torch.from_numpy(exposure3).permute(2, 0, 1).unsqueeze(0)

    hdr_image = torch.clip(torch.from_numpy(hdr_image).permute(2, 0, 1).unsqueeze(0), 0, 1)
    
    ssim1 = ssim(exposure1, hdr_image).item()
    ssim2 = ssim(exposure2, hdr_image).item()
    ssim3 = ssim(exposure3, hdr_image).item()
    print(i, (ssim1 + ssim2 + ssim3)/3)
    total_ssim += (ssim1 + ssim2 + ssim3)
    
    psnr1 = psnr(exposure1[0], hdr_image[0]).item()
    psnr2 = psnr(exposure2[0], hdr_image[0]).item()
    psnr3 = psnr(exposure3[0], hdr_image[0]).item()
    print(i, (psnr1 + psnr2 + psnr3)/3)
    total_psnr += (psnr1 + psnr2 + psnr3)
    
total_ssim /= (3 * data.shape[0])
print("SSIM", total_ssim)
total_psnr /= (3 * data.shape[0])
print("PSNR", total_ssim)
