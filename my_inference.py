import argparse
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb

# data libraries 

from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np






# read the images for the first test
name = './sequences-train/swan' # name of the sequence : bag, bear, book, camel, rhino, swan
im_begin, im_end = 1, 2 # reference image (1) and last image (varies depending on the sequence)

im = 1
mask_begin  = io.imread(name+'-%0*d.png'%(3,im)) 
img_begin = io.imread(name+'-%0*d.bmp'%(3,im)) 

im = 20
mask_end = io.imread(name+'-%0*d.png'%(3,im)) 
img_end  = io.imread(name+'-%0*d.bmp'%(3,im)) 

# Compute the difference between the two masks



# Plot the images and masks
plt.subplot(2,4,1)
plt.imshow(img_begin)
plt.title('Image Begin')
plt.axis('off')

plt.subplot(2,4,2)
plt.imshow(mask_begin, cmap='gray')
plt.title('Mask Begin')
plt.axis('off')

plt.subplot(2,4,3)
plt.imshow(img_end)
plt.title('Image End')
plt.axis('off')

plt.subplot(2,4,4)
plt.imshow(mask_begin, cmap='gray')
plt.imshow(mask_end, cmap='gray', alpha=0.5)
plt.title('Mask End')
plt.axis('off')

plt.show()