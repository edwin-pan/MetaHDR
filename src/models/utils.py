from src.models.UNet import get_unet

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