#!/usr/bin/env python
import os
import sys
from PIL import Image

im1 = Image.open(sys.argv[1])
im2 = Image.open(sys.argv[2])

size, mode = im2.size, im2.mode

with Image.new(mode, size) as f:
  out, in1, in2 = f.load(), im1.load(), im2.load()
  for i in range(size[0]):
    for j in range(size[1]):
      out[i, j] = in2[i, j] if in1[i, j] != in2[i, j] else (0, 0, 0, 0)
  f.save('ans_two.png', 'PNG')
