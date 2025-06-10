import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def show_imgs(x, title=None, save_path=None):
    """显示图像网格"""
    plt.figure(figsize=(10, 10))
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1)
    plt.imshow(grid.numpy())
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
