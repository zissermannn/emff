import numpy as np
import torch
import scipy.stats as st
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models.video as models
import torch.nn as nn
# from tools import clip_by_tensor, normalize_torchimage, Normalize
from torchvision.utils import save_image
from os.path import join,exists
import os

class Depth_wise_Conv:
    def __init__(self,device,kern_len = 7,n_sig = 3) -> None:
        self.device =device
        self.kern_len = kern_len
        self.n_sig = n_sig
        self.kernel = self.get_kernel()
        # self.stack_kern, self.padding_size_s = project_kern_3D(kern_size_s,kern_size_t, sample_std)
        # self.stack_kern = self.stack_kern.to(device,non_blocking=True)

    def get_kernel(self) -> torch.Tensor:
        kernel = self.gkern(self.kern_len, self.n_sig).astype(np.float32)

        kernel = np.expand_dims(kernel, axis=0)  # (W, H) -> (1, W, H)
        kernel = np.repeat(kernel, 3, axis=0)  # -> (C, W, H)
        kernel = np.expand_dims(kernel, axis=1)  # -> (C, 1, W, H)
        return torch.from_numpy(kernel).to(self.device)

    @staticmethod
    def gkern(kern_len: int = 15, n_sig: int = 3) -> np.ndarray:
        """Return a 2D Gaussian kernel array."""

        interval = (2 * n_sig + 1.0) / kern_len
        x = np.linspace(-n_sig - interval / 2.0, n_sig + interval / 2.0, kern_len + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    
    def __call__(self, x):
        x = F.conv2d(x, self.kernel, stride=1, padding="same", groups=3)
        return x

def input_diversity(
    x: torch.Tensor, image_resize: int = 331, diversity_prob: float = 0.5
) -> torch.Tensor:
    """Apply input diversity to a batch of images.

    Note:
        Adapted from the TensorFlow implementation (cihangxie/DI-2-FGSM):
        https://github.com/cihangxie/DI-2-FGSM/blob/10ffd9b9e94585b6a3b9d6858a9a929dc488fc02/attack.py#L153-L164

    Args:
        x: A batch of images. Shape: (N, C, H, W).
        resize_rate: The resize rate. Defaults to 0.9.
        diversity_prob: Applying input diversity with probability. Defaults to 0.5.

    Returns:
        The diversified batch of images. Shape: (N, C, H, W).
    """

    if torch.rand(1) < diversity_prob:
        return x

    img_size = x.shape[-1]
    img_resize = image_resize
    # if resize_rate < 1:
    #     img_size = img_resize
    #     img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode="nearest")

    h_rem = img_resize - rnd
    w_rem = img_resize - rnd

    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    pad = [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()]
    padded = F.pad(rescaled, pad=pad, mode="constant", value=0)
    padded = F.interpolate(padded, size=[img_size, img_size], mode="nearest")
    return padded

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def TIM(surrogate_model,images, labels, args,num_iter = 10):
    print(TIM.__name__)
    eps = args.eps / 255.0
    alpha = eps / num_iter
    image_min = clip_by_tensor(images - eps, 0.0, 1.0)
    image_max = clip_by_tensor(images + eps, 0.0, 1.0)
    depwise_conv = Depth_wise_Conv(images.device,kern_len=15)
    grad = 0
    momentum = 1.0
    for i in range(num_iter):
        # zero_gradients(x)
        if images.grad is not None:
            images.grad.zero_()
        images = Variable(images, requires_grad = True)
        out_logits = surrogate_model(images)
        if type(out_logits) is list:
            logits = out_logits[0]
            aux_logits = out_logits[1]
        else:
            logits = out_logits
        _ , indices = torch.sort(logits)
        back_loss = F.cross_entropy(logits, labels)
        if type(out_logits) is list:
            back_loss += F.cross_entropy(aux_logits, labels)
        back_noise = torch.autograd.grad(back_loss,images)[0]
        noise = back_noise
        noise = depwise_conv(noise)
        noise = noise / torch.mean(torch.abs(noise), dim=[1, 2, 3], keepdims=True)
        grad = momentum * grad + noise

        images = images + alpha * torch.sign(grad)
        images = clip_by_tensor(images, image_min, image_max)
    
    # if exists(join(args.image_save_root,'TIM')) is False:
    #     os.makedirs(join(args.image_save_root,'TIM'))
    # for i in range(images.size()[0]): 
    #     save_image(images[i].cpu(),join(args.image_save_root,'TIM',str(args.image_num).rjust(4,'0')+'.png'))   
    #     args.image_num += 1
    return images

def DIM(surrogate_model,images, labels, args,num_iter = 10):
    print(DIM.__name__)
    eps = args.eps/ 255.0
    alpha = eps / num_iter
    image_min = clip_by_tensor(images - eps, 0.0, 1.0)
    image_max = clip_by_tensor(images + eps, 0.0, 1.0)
    # depwise_conv = Depth_wise_Conv(images.device,kern_len=15)
    grad = 0
    momentum = 1.0
    for i in range(num_iter):
        if images.grad is not None:
            images.grad.zero_()
        images = Variable(images, requires_grad = True)
        out_logits = surrogate_model(input_diversity(images))
        if type(out_logits) is list:
            logits = out_logits[0]
            aux_logits = out_logits[1]
        else:
            logits = out_logits
        back_loss = F.cross_entropy(logits, labels)
        if type(out_logits) is list:
            back_loss += F.cross_entropy(aux_logits, labels)
        back_noise = torch.autograd.grad(back_loss,images)[0]
        noise = back_noise
        # noise = depwise_conv(noise)
        noise = noise / torch.mean(torch.abs(noise), dim=[1, 2, 3], keepdims=True)
        grad = momentum * grad + noise

        images = images + alpha * torch.sign(grad)
        images = clip_by_tensor(images, image_min, image_max)
    # if exists(join(args.image_save_root,'DIM')) is False:
    #     os.makedirs(join(args.image_save_root,'DIM'))
    # for i in range(images.size()[0]): 
    #     save_image(images[i].cpu(),join(args.image_save_root,'DIM',str(args.image_num).rjust(4,'0')+'.png'))   
    #     args.image_num += 1
    return images