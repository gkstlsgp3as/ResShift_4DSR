import os
import cv2
from pathlib import Path
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Resize the input")
	parser.add_argument('-i', '--input',
						type=str,
						required=False,
						default='../data/sample_retina_org/',
						help='Input image path')
	parser.add_argument('-o', '--output',
						type=str,
						required=False,
						default='../data/sample_retina/',
						help = 'Output image path after resizing')
	opt = parser.parse_args()
	fl_list = os.listdir(opt.input)
	os.makedirs(opt.output, exist_ok=True)
	for f in fl_list:
		im = cv2.imread(Path(opt.input, f))
		dst = cv2.resize(im, (64, 64), fx=4, fy=4, interpolation=cv2.INTER_AREA)
		cv2.imwrite(Path(opt.output, f), dst)