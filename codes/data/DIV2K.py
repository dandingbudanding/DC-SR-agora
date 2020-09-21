import torch.utils.data as data
import os.path
from PIL import Image
import numpy as np
from data import common
import cv2
import PIL.Image as pil_image

def default_loader(path):
    return np.array(pil_image.open(path).convert('RGB'))
    # return cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = [
    '.JPG', '.jpg','.png',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class div2k(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext   # '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        self.root = './datasets/train_data/'
        self.dir_hr = os.path.join(self.root, 'HR_Down2')
        self.dir_lr = os.path.join(self.root, 'LR_Down2')

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)

        # prob = 1.0
        # alpha = 0.7
        # aux_prob = 1.0
        # aux_alpha = 1.2
        # hr_tensor, lr_tensor = common.cutmixup(
        #     hr_tensor.clone(), lr_tensor.clone(),
        #     mixup_prob=aux_prob, mixup_alpha=aux_alpha,
        #     cutmix_prob=prob, cutmix_alpha=alpha,
        # )
        # hr_=tensor_to_np(hr_tensor)
        # lr_=tensor_to_np(lr_tensor)
        #
        # cv2.imwrite("./hr.png",hr_)
        # cv2.imwrite("./lr.png", lr_)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)

        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img