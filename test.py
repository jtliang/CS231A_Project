import numpy as np
import tensorflow as tf
import sqlite3
from random import random
from PIL import Image, ImageFilter
import os
import cv2
import matplotlib.pyplot as plt


# conn = sqlite3.connect('train.db')
# c = conn.cursor()

# start = 1584

# curs = c.execute('SELECT id FROM train where id < 1585').fetchall()
# for i in curs:
#   t = int(i[0])
#   items = []
  # label = c.execute('SELECT species FROM train WHERE id=' + str(t))
  # label = int(label.next()[0])
image_file = Image.open("images/" + str(537) + ".jpg")
image_file = image_file.convert('1')
original = image_file
width, height = image_file.size
maxwh = width if width > height else height
scale = 1633 / float(maxwh)
image_file = image_file.resize((int(image_file.size[0] * scale), int(image_file.size[1] * scale)))
print(scale)
bw_im = Image.new('1', (1633, 1633), 0)
arr = np.array(image_file)
targ_x = int(1633/2 - arr.shape[1]/2)
targ_y = int(1633/2 - arr.shape[0]/2)
bw_im.paste(image_file, (targ_x, targ_y))
bw_im = bw_im.resize((100, 100))



image10 = bw_im.rotate(10, expand=True)
# imageNeg10 = bw_im.rotate(-10, expand=True)
bw_im = bw_im.filter(ImageFilter.FIND_EDGES)

image_file = image_file.filter(ImageFilter.FIND_EDGES)
image10 = image10.filter(ImageFilter.FIND_EDGES)
# imageNeg10 = imageNeg10.filter(ImageFilter.FIND_EDGES)

# plt.imshow(np.array(original), cmap='Greys')
# plt.imshow(np.array(image_file), cmap='Greys')
# plt.imshow(np.array(bw_im), cmap='Greys')
plt.imshow(np.array(image10), cmap='Greys')
# plt.imshow(np.array(image_file), cmap='Greys')
plt.show()
exit()
# conn.commit()
# conn.close()