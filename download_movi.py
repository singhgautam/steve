import os
import argparse

import tensorflow_datasets as tfds
import torchvision.utils as vutils

from torchvision import transforms


parser = argparse.ArgumentParser()

parser.add_argument('--out_path', default='data/')
parser.add_argument('--level', default='e')
parser.add_argument('--split', default='train')
parser.add_argument('--version', default='1.0.0')
parser.add_argument('--image_size', type=int, default=128)

args = parser.parse_args()

ds, ds_info = tfds.load(f"movi_{args.level}/{args.image_size}x{args.image_size}:{args.version}", data_dir="gs://kubric-public/tfds", with_info=True)
train_iter = iter(tfds.as_numpy(ds[args.split]))

to_tensor = transforms.ToTensor()

b = 0
for record in train_iter:
    video = record['video']
    T, *_ = video.shape

    # setup dirs
    path_vid = os.path.join(args.out_path, f"{b:08}")
    os.makedirs(path_vid, exist_ok=True)

    for t in range(T):
        img = video[t]
        img = to_tensor(img)
        vutils.save_image(img, os.path.join(path_vid, f"{t:08}_image.png"))

    b += 1
