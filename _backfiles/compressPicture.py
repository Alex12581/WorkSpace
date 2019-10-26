#!/usr/bin/python3

from PIL import Image
import sys

path = sys.argv[1]

with Image.open(path, 'r') as img:
    x, y = img.size
    img.resize((x//2, y//2)).save(path[:-4]+"_compressed"+path[-4:])
