from PIL import Image
import numpy as np
import sys
im= np.array(Image.open("D:\College\Project\Screenshot_1.jpg"))
np.set_printoptions(threshold=sys.maxsize)
print(im)