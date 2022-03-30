import numpy as np
import cv2
from matplotlib import pyplot as plt
%matplotlib inline

img = cv2.imread('wt.jpg')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (150,60,600,566)
cv2.grabCut(img,mask,rect,
            bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.xticks([])
plt.yticks([])
plt.show()
#=====================================
# newmask is the mask image I manually labelled
newmask = cv2.imread('wtmask.jpg',0)

# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1

cv2.grabCut(img,mask,None,
            bgdModel,fgdModel,
            5,cv2.GC_INIT_WITH_MASK)

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]

plt.figure(figsize=(18,9))

plt.subplot(121)
plt.imshow(mask),plt.xticks([]),
plt.yticks([])

plt.subplot(122)
plt.imshow(img),plt.xticks([]),
plt.yticks([])

plt.show()