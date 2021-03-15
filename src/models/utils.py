import torch
from torch import nn
from src.models.UNet import get_unet


# Define Network Blocks
def convolution_block(in_dim, out_dim):
    return nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1)

def transpose_convolution_block(in_dim, out_dim):
    return torch.nn.ConvTranspose2d(in_dim,out_dim,kernel_size=3, stride=2, padding=1,output_padding=1)

def contract_block(in_dim,out_dim,activation):
    block = nn.Sequential(convolution_block(in_dim, out_dim),
                            nn.BatchNorm2d(out_dim),
                            activation,
                            convolution_block(out_dim, out_dim),
                            nn.BatchNorm2d(out_dim),
                            activation)
    return block

def bottom_block(in_dim, out_dim, activation):
    bottom = torch.nn.Sequential(convolution_block(in_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 convolution_block(out_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 transpose_convolution_block(out_dim, in_dim),
                                 nn.BatchNorm2d(in_dim),
                                 activation)
    return bottom

def expand_block(in_dim, out_dim, activation):
    expand = torch.nn.Sequential(convolution_block(in_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 convolution_block(out_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 transpose_convolution_block(out_dim, out_dim//2),
                                 nn.BatchNorm2d(out_dim//2),
                                 activation)
    return expand

def top_block(in_dim, out_dim, activation):
    top = torch.nn.Sequential(convolution_block(in_dim, out_dim),
                              nn.BatchNorm2d(out_dim),
                              activation,
                              convolution_block(out_dim, out_dim),
                              # nn.BatchNorm2d(out_dim),
                              # activation,
                              # convolution_block(out_dim,NUM_CLASSES),
                              # nn.BatchNorm2d(NUM_CLASSES),
                              # activation
                              )
    return top







def copy_model_fn(mi,width,height):
    copied_model = get_unet(width,height)
    copied_model.set_weights(mi.get_weights())
    return copied_model

def save_model(self, performance, epoch):
    # TODO: Edit to work with tensorflow. Unused atm
    save_dict = {
        'epoch': epoch,
        'gen_state_dict': self.generator.state_dict(),
        'performance': performance,
        'gen_optimizer': self.gen_optimizer.state_dict(),
        'disc_motion_state_dict': self.motion_discriminator.state_dict(),
        'disc_motion_optimizer': self.dis_motion_optimizer.state_dict(),
    }

    filename = osp.join(self.logdir, 'checkpoint.pth.tar')
    torch.save(save_dict, filename)

    if self.performance_type == 'min':
        is_best = performance < self.best_performance
    else:
        is_best = performance > self.best_performance

    if is_best:
        logger.info('Best performance achieved, saving it!')
        self.best_performance = performance
        shutil.copyfile(filename, osp.join(self.logdir, 'model_best.pth.tar'))

        with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
            f.write(str(float(performance)))