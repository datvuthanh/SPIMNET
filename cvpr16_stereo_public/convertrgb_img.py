import cv2
import numpy as np

max_disparity = 52
scale_factor = 255 / (max_disparity - 1) 
 



import glob
path = '/home/memi/Desktop/SPIMNET/results_torch/220951_043015'
for filename in glob.glob(path + '/*.png'): 
    
    print(filename)
    img = cv2.imread(filename,0) * scale_factor

    img = img.astype(np.uint8) 

    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    cv2.imwrite(filename,img)
