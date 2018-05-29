#!/usr/bin/env python

from PIL import Image

# downsample images to 224x224
im1_col = Image.open('n02099601_634_col.JPEG')
im2_col = Image.open('n03792782_1155_col.JPEG')
im3_col = Image.open('n04505470_10690_col.JPEG')

im1_col.thumbnail((224, 224),
                  Image.ANTIALIAS)
im1_col.save('n02099601_634_224x224.JPEG', 'JPEG')
im2_col.thumbnail((224, 224),
                  Image.ANTIALIAS)
im2_col.save('n03792782_1155_224x224.JPEG', 'JPEG')
im3_col.thumbnail((224, 224),
                  Image.ANTIALIAS)
im3_col.save('n04505470_10690_224x224.JPEG', 'JPEG')

