# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:46:42 2024

@author: hskim
"""

from pathlib import Path 
import cv2
import numpy as np


retinapath = Path('/home/data/4DRADAR/ResShift/data/traindata/retina/lq')
savepath = Path('/home/data/4DRADAR/ResShift/data/traindata/retina/mask_17')

savepath.mkdir(exist_ok=True, parents=True)

kernel = 17

retinaimglist = list(retinapath.glob("*.png"))

for rimgpath in retinaimglist:
    
    rimg = cv2.imread(str(rimgpath))
    
    rimg_gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    
    rimg_b = np.where(rimg_gray>0, 1, 0).astype(np.uint8)
    
    dst = cv2.distanceTransform(rimg_b, cv2.DIST_L2, 5)
    dst = (dst/(dst.max()-dst.min()) * 255).astype(np.uint8)
    
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel,kernel))
    tophat = cv2.morphologyEx(dst, cv2.MORPH_TOPHAT, k)
    tophat_m = np.where(tophat>0, 0, 255)
    
    # plt.subplot(131); plt.imshow(rimg); plt.axis('off')
    # plt.subplot(132); plt.imshow(dst); plt.axis('off')
    # plt.subplot(133); plt.imshow(tophat_m); plt.axis('off')
    # plt.show()
    
    maskpath = savepath / (rimgpath.stem + f"_mask{kernel}.png")
    cv2.imwrite(str(maskpath), tophat_m)

