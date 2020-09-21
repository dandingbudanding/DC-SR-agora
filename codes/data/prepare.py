import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from torchvision.transforms import transforms


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    image_list = sorted(glob.glob('{}/*'.format(args.images_dir+"//HR")))
    patch_idx = 0

    for i, image_path in enumerate(image_list):
        hr = pil_image.open(image_path).convert('RGB')
        lr_path=image_path[:-13]+  "//LR"+image_path[-9:]
        lr = pil_image.open(lr_path).convert('RGB')

        hr = np.array(hr)
        lr = np.array(lr)

        for hr in transforms.FiveCrop(size=(hr.height // 2, hr.width // 2))(hr):
            hr = np.array(hr)
            lr = np.array(lr)

            lr_group.create_dataset(str(patch_idx), data=lr)
            hr_group.create_dataset(str(patch_idx), data=hr)

            patch_idx += 1

        print(i, patch_idx, image_path)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)

        hr = np.array(hr)
        lr = np.array(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

        print(i)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=False,default="D:\\WFY\\20200628SR\\20200806miniSR\\agora_sr\\train")
    parser.add_argument('--output-path', type=str, required=False,default="./h5file_train_HR_x2")
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true',default=False)
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
