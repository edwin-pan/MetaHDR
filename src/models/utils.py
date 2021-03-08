from src.models.UNet import get_unet

def copy_model_fn(mi,width,height):
    copied_model = get_unet(width,height)
    copied_model.set_weights(mi.get_weights())
    return copied_model