import os
import cv2
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate reversed masks for input retina")
    parser.add_argument('-i', '--input',
                        type=str,
                        required=False,
                        default='../data/sample_retina/',
                        help = 'Input image path')
    parser.add_argument('-o', '--output',
                        type=str,
                        required=False,
                        default='../data/sample_retina_mask/',
                        help = 'Output image path after resizing')
    opt = parser.parse_args()

    fl_list = os.listdir(opt.input)
    os.makedirs(opt.output, exist_ok=True)
    for f in fl_list:
        im = cv2.imread(Path(opt.input, f))
        mask = 255 - im;
        mask = (mask == 255) * 255
        mask = 0.2126 * mask[..., 0] + 0.7152 * mask[..., 1] + 0.0722 * mask[..., 2]
        cv2.imwrite(Path(opt.output, f), mask)