import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import argparse
from skimage import transform

parser = argparse.ArgumentParser()
parser.add_argument('--org', type=str, default='data/content/01.jpg')
parser.add_argument('--target', type=str, default='output/01_01.png')
args = parser.parse_args()

def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr

image1 = cv2.imread(args.org)
image2 = cv2.imread(args.target)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

if image1_gray.shape != image2_gray.shape:
    image1_gray = transform.resize(image1_gray, image2_gray.shape, anti_aliasing=True)
    image1_gray = (image1_gray * 255).astype(np.uint8)

(score, diff) = ssim(image1_gray, image2_gray, full=True)
print(f"SSIM: {score}")

psnr = compute_PSNR(image1_gray, image2_gray)
print(f"PSNR: {psnr} dB")