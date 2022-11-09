import os
import glob
import torch

from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobVideoDataset(Dataset):
    def __init__(self, root, phase, img_size, ep_len=3, img_glob='*.png'):
        self.root = root
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        else:
            pass

        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video


class GlobVideoDatasetWithMasks(Dataset):
    def __init__(self, root, img_size, num_segs, ep_len=6, img_glob='*_image.png'):
        self.root = root
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        # chunk into episodes
        self.episodes_rgb = []
        self.episodes_mask = []
        for dir in self.total_dirs:
            frame_buffer = []
            mask_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for image_path in image_paths:
                p = Path(image_path)

                frame_buffer.append(p)
                mask_buffer.append([
                    p.parent / f"{p.stem.split('_')[0]}_mask_{n:02}.png" for n in range(num_segs)
                ])

                if len(frame_buffer) == self.ep_len:
                    self.episodes_rgb.append(frame_buffer)
                    self.episodes_mask.append(mask_buffer)
                    frame_buffer = []
                    mask_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes_rgb)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes_rgb[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)

        masks = []
        for mask_locs in self.episodes_mask[idx]:
            frame_masks = []
            for mask_loc in mask_locs:
                image = Image.open(mask_loc).convert('1')
                image = image.resize((self.img_size, self.img_size))
                tensor_image = self.transform(image)
                frame_masks += [tensor_image]
            frame_masks = torch.stack(frame_masks, dim=0)
            masks += [frame_masks]
        masks = torch.stack(masks, dim=0)

        return video, masks
