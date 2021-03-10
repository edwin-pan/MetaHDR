import numpy as np
from skimage.filters import gaussian

def fspecial_gaussian_2d(size, sigma):
    kernel = np.zeros(tuple(size))
    kernel[size[0]//2, size[1]//2] = 1
    kernel = gaussian(kernel, sigma)
    return kernel/np.sum(kernel)

def bilateral2d(img, radius, sigma, sigmaIntensity):
    pad = radius
    # Initialize filtered image to 0
    out = np.zeros_like(img)

    # Pad image to reduce boundary artifacts
    imgPad = np.pad(img, pad)

    # Smoothing kernel, gaussian with standard deviation sigma
    # and size (2*radius+1, 2*radius+1)
    filtSize = (2*radius + 1, 2*radius + 1)
    spatialKernel = fspecial_gaussian_2d(filtSize, sigma)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerVal = imgPad[y+pad, x+pad] # Careful of padding amount!

            # Go over a window of size (2*radius + 1) around the current pixel,
            # compute weights, sum the weighted intensity.
            # Don't forget to normalize by the sum of the weights used.
            windowed_intensities = imgPad[y:y+2*pad+1, x:x+2*pad+1]
            gaussian_intensities = np.exp((-1*np.abs(windowed_intensities-centerVal)**2)/(2*sigmaIntensity**2))
            weights = spatialKernel*gaussian_intensities
            weighted_intensities = windowed_intensities*weights
            
            out[y, x] = np.sum(weighted_intensities)/np.sum(weights)
    return out


