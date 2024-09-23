from tifffile import imwrite
import os
import cv2 
from pathlib import Path
import argparse

MIN = 0; MAX = 5

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Unnormlize the input")
	parser.add_argument('-i', '--input',
						type=str,
						required=False,
						default='../results/inpaint/',
						help='Input image path')
	parser.add_argument('-o', '--output',
						type=str,
						required=False,
						default='../results/inpaint_unnorm/',
						help = 'Output image path after unnormlization')
	opt = parser.parse_args()
	fl_list = os.listdir(opt.input)
	os.makedirs(opt.output, exist_ok=True)
	os.makedirs(opt.output.replace('inpaint','sr'), exist_ok=True)
	for f in fl_list:
		inp_im = cv2.imread(Path(opt.input, f))
		inp_im_unnorm = (inp_im/255 * (MAX - MIN)) + MIN
		imwrite(Path(opt.output, f), inp_im_unnorm)
		
		sr_im = cv2.imread(Path(opt.input, f))
		sr_im_unnorm = (sr_im/255 * (MAX - MIN)) + MIN
		imwrite(Path(opt.output.replace('inpaint','sr'), f), sr_im_unnorm)
