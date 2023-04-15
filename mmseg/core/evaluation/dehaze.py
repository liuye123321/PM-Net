import os
import cv2
import numpy as np
import torch
import math
import mmcv
from skimage import measure
def check():
    import subprocess
    import sys
    FNULL = open(os.devnull, 'w')
    result = subprocess.call(
        '/home/ext2/liuye/lane-detection/resa/runner/evaluator/culane/lane_evaluation/evaluate', stdout=FNULL, stderr=FNULL)
    if result > 1:
        print('There is something wrong with evaluate tool, please compile it.')
        sys.exit()

def calc_psnr(img1, img2):
    # img1 = img1[0].cpu().float().numpy()
    # img2 = img2[0].cpu().float().numpy()
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def save(output, out_dir, img_name):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    if out_dir is None:
        out_dir = './test_result/'
        if not os.path.exists(out_dir):
           os.mkdir(out_dir)
    # print(os.path.abspath(out_dir))
    if 'dehaze' in output:
        dehaze = output['dehaze'].cpu().numpy()[0]
        dehaze = numpy2imgs(dehaze, mean, std)
        gt = output['gt'].cpu().numpy()[0]
        gt = numpy2imgs(gt, mean, std)
        outname = out_dir + img_name[40:]
        outdir = os.path.dirname(outname)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        cv2.imwrite(outname, dehaze)
    psnr = calc_psnr(dehaze, gt)
    ssim = measure.compare_ssim(dehaze, gt, data_range=255, multichannel=True)

    return psnr, ssim

def quantize(img, rgb_range=255):
    pixel_range = 255 / rgb_range
    return (img*pixel_range).clip(0, 255).round()/pixel_range

def numpy2imgs(numpy, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to 3-channel images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError('pytorch is not installed')
    # assert torch.is_tensor(numpy) and numpy.ndim == 3
    assert len(mean) == 3
    assert len(std) == 3
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []

    img = numpy.transpose(1, 2, 0)
    # img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb).clip(0, 255).astype(np.uint8)
    img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
    imgs.append(np.ascontiguousarray(img))
    return img

if __name__ == '__main__':
    img = torch.ones((1,1))*(-300)
    a = quantize(img)
    b=1