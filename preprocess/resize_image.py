from __future__ import print_function
import os
from wand.image import Image

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    for root, dirs, files in os.walk('data'):
        for name in files:
            tokens = root.split('/')
            mil = int(tokens[1])
            tho = int(tokens[2])
            one = int(tokens[3])
            image_id = mil * 1000000 + tho * 1000 + one
            image_path = os.path.join(root, name)
            with Image(filename=image_path) as img:
                img.transform(resize='128x128^')
                img.crop(width=128, height=128, gravity='center')
                img.format = 'png'
                img.save(filename=("output/%07d.png" % image_id))
