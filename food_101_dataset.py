import json
import os
import tarfile
from pathlib import Path

import numpy as np
import requests
from albumentations import Compose
from albumentations.augmentations import transforms
from chainer.dataset import DatasetMixin
from PIL import Image
from tqdm import tqdm


class Food101Dataset(DatasetMixin):

    def __init__(self, data_dir='auto',
                 train=True, augmentation=None,
                 index=None, drop_index=None,):
        assert index is None or drop_index is None

        if data_dir == 'auto':
            dataset_dir = Path('./dataset')
            if not dataset_dir.exists():
                dataset_dir.mkdir()

            self.data_dir = dataset_dir / 'food-101'
            if not self.data_dir.exists():
                url = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
                tar_file = dataset_dir / 'food-101.tar.gz'

                file_size = int(requests.head(url).headers["content-length"])
                res = requests.get(url, stream=True)
                if res.status_code == 200:
                    with tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
                        with open(tar_file, 'wb') as f:
                            for chunk in res.iter_content(chunk_size=1024):
                                f.write(chunk)
                                pbar.update(len(chunk))
                with tarfile.open(tar_file, 'r:gz') as tf:
                    tf.extractall(path=dataset_dir)
                tar_file.unlink()

        self.imgs = []
        self.labels = []
        json_file = 'train.json' if train else 'test.json'
        with open(self.data_dir / 'meta' / json_file, 'r') as f:
            data = json.load(f)

        for i, (k, v) in enumerate(sorted(data.items())):
            self.imgs.extend(data[k])
            self.labels.extend([i] * len(data[k]))

        if train:
            NG_list = ['bread_pudding/1375816', 'lasagna/3787908', 'steak/1340977']
            for ng_img in NG_list:
                ng_index = self.imgs.index(ng_img)
                del self.imgs[ng_index]
                del self.labels[ng_index]

        if index is not None:
            self.imgs = self.imgs[index]
            self.labels = self.labels[index]
        elif drop_index is not None:
            del self.imgs[drop_index]
            del self.labels[drop_index]

        if augmentation is not None:
            processes = []
            for process, params in augmentation.items():
                processes.append(getattr(transforms, process)(**params))
            self.augmentation = Compose(processes, p=1.0)
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.imgs)

    def get_example(self, i):
        img_path = self.data_dir / 'images' / '{}.jpg'.format(self.imgs[i])
        img = np.array(Image.open(img_path), np.float32)
        label = np.int32(self.labels[i])

        if self.augmentation is not None:
            img = self.augmentation(image=img)['image']
            if img.shape != (224, 224, 3):
                print(img.shape, self.imgs[i])
            img = img.transpose((2, 0, 1))
        img = img / 127.5 - 1.0

        return img, label


if __name__ == '__main__':

    augmentation = {
        'HorizontalFlip': {'p': 0.5},
        'VerticalFlip': {'p': 0.5},
        'PadIfNeeded': {'p': 1.0, 'min_height': 512, 'min_width': 512},
        'Rotate': {'p': 1.0, 'limit': 45, 'interpolation': 1},
        'Resize': {'p': 1.0, 'height': 248, 'width': 248, 'interpolation': 2},
        'RandomScale': {'p': 1.0, 'scale_limit': 0.09, 'interpolation': 2},
        'RandomCrop': {'p': 1.0, 'height': 224, 'width': 224},
        'Cutout': {'p': 0.5, 'num_holes': 8, 'max_h_size': 8, 'max_w_size': 8}
    }
    resize = {
        'PadIfNeeded': {'p': 1.0, 'min_height': 512, 'min_width': 512},
        'Resize': {'p': 1.0, 'height': 224, 'width': 224, 'interpolation': 2}
    }

    train_data = Food101Dataset(augmentation=resize)
    valid_data = Food101Dataset(train=False, augmentation=resize)
    print(len(train_data), len(valid_data))
