import numpy as np
from src.dataset.hdr_utils import bilateral2d

def _do_gamma_correction(img, gamma):
    return np.clip(img ** (1/gamma), 0, 1)

def _do_local_tone_mapping(img, gamma, averageFilterRadius=5, sigma=3, sigmaIntensity=0.4, dR=1.6):
    # Intensity
    img = np.clip(img, 0, 1)
    intensity = (20 * img[..., 0] + 40 * img[..., 1] + img[..., 2]) / 61
    
    # Chrominance channels
    chrominance = np.zeros_like(img) 
    chrominance[:, :, 0] = img[:, :, 0] / intensity
    chrominance[:, :, 1] = img[:, :, 1] / intensity
    chrominance[:, :, 2] = img[:, :, 2] / intensity
        
    # Perceptually linear space
    L = np.log10(intensity)

    # Bilateral filter on L
    averageFilterRadius = averageFilterRadius
    sigma = sigma
    sigmaIntensity = sigmaIntensity
    B = bilateral2d(L, averageFilterRadius, sigma, sigmaIntensity)
    D = L - B
    
    # Correct base layer
    o = np.max(B)
    dR = dR
    s = dR / (np.max(B) - np.min(B))
    BB = (B - o) * s
    
    scaled_intensity = 10 ** (BB+D)
    tone_mapped = np.zeros_like(img)
    tone_mapped[:, :, 0] = scaled_intensity * chrominance[:, :, 0]
    tone_mapped[:, :, 1] = scaled_intensity * chrominance[:, :, 1]
    tone_mapped[:, :, 2] = scaled_intensity * chrominance[:, :, 2]
    return np.clip(((tone_mapped) ** 1/gamma), 0, 1)


def visualize_hdr_image(img, method="gamma_correct", gamma=2.2):
    """
    Returns a tone mapped or gamma corrected hdr img
    Args:
        - img: input hdr img
        - method: either "gamma_correct" or "tone_map"
        - gamma: gamma correction factor to use
    """
    if method == "gamma_correct":
        return _do_gamma_correction(img, gamma)
    elif method == "tone_map":
        return _do_local_tone_mapping(img, gamma)
    else:
        raise NotImplementedError
        
    