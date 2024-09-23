import os
import cv2
from pathlib import Path

path = '/home/data/4DRADAR/ResShift/results/retina5cm_mask0_infer_test/'
dstpath = '/home/data/4DRADAR/ResShift/results/retina5cm_mask0_infer_test0/'
fl_list = os.listdir(path)
for f in fl_list:
	im = cv2.imread(Path(path, f))
	dst = cv2.resize(im, (64, 64), fx=4, fy=4, interpolation=cv2.INTER_AREA)
	cv2.imwrite(Path(dstpath, f), dst)