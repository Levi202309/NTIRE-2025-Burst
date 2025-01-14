from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage import io
import numpy as np
import sys
import re
import cv2

### for debug
def list_files(startpath, file=sys.stdout):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if 'tmp' in f or 'metadata' in f:
                continue
            print('{}{}'.format(subindent, f), file=file)


def calculate_score(out, gt):

    # calculate metrics on rgb domain
    psnr_v = psnr(gt, out)
    ssim_v = ssim(rgb2gray(gt), rgb2gray(out))

    return psnr_v, ssim_v


def main(input_dir, output_dir):
    input_folder = os.path.join(input_dir, 'res')
    if len(os.listdir(input_folder)) == 1:
        input_folder = os.path.join(input_folder, os.listdir(input_folder)[0])
    gt_folder = os.path.join(input_dir, 'ref')
    if len(os.listdir(gt_folder)) == 1:
        gt_folder = os.path.join(gt_folder, os.listdir(gt_folder)[0])

    output_filename = os.path.join(output_dir, 'scores.txt')

    total_status = [0, 0]
    total_count = 0
    # lines = []
    for path in sorted(glob(f'{input_folder}/*.tif')):
        gt_name=os.path.basename(path).replace('out','gt')
        gt_im = io.imread(f'{gt_folder}/{gt_name}')
        gt_im = np.clip(gt_im * 255, 0, 255).astype(np.uint8)
        out = io.imread(path)
        psnr_v, ssim_v = calculate_score(out, gt_im)
        total_status[0] += psnr_v
        total_status[1] += ssim_v

    
        total_count += 1

    with open(output_filename, 'w') as out_fp:
        out_fp.write('{}: {}\n'.format('PSNR', total_status[0] / total_count))
        out_fp.write('{}: {}\n'.format('SSIM', total_status[1] / total_count))
        out_fp.write('{}: {}\n'.format('Score', total_status[0] / total_count))
        out_fp.write('DEVICE: CPU')
        # out_fp.write('DETAILS:\n')
        # out_fp.writelines(lines)


if __name__ == '__main__':
    try:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        main(input_dir, output_dir)
    except Exception as e:
        print('Detailed files:', file=sys.stderr)
        list_files(input_dir, file=sys.stderr)
        print("", file=sys.stderr)
        raise e
