# copyright by dandingbudanding
import cv2
import os
from utils import utils as utils


def dataprocessing():
# data cleaning
    print("start data cleaning")
    HR_path = "./datasets/train_data/HR"
    HR_down2_path = "./datasets/train_data/HR_Down2"
    LR_path = "./datasets/train_data/LR"
    LR_down2_path = "./datasets/train_data/LR_Down2"

    path_list = os.listdir(LR_path)
    if not os.path.exists(HR_down2_path):
        os.mkdir(HR_down2_path)

    if not os.path.exists(LR_down2_path):
        os.mkdir(LR_down2_path)

    count = 0
    for file in path_list:
        lr_img = cv2.imread(os.path.join(LR_path, file), -1)
        h, w = lr_img.shape[:2]

        hr_img = cv2.imread(os.path.join(HR_path, file), -1)

        train_hr_img = cv2.resize(hr_img, (w, h), interpolation=cv2.INTER_LINEAR)

        ssim_ = utils.compute_ssim(train_hr_img, lr_img)
        if ssim_ < 0.4:  # pass not match
            print(file + " is mismatch!")
            continue

        if h < 500 or w < 500:  # too small not downsampling
            print(file, h, w)
            cv2.imwrite(os.path.join(HR_down2_path, str(count)+file[-4:]), hr_img)
            cv2.imwrite(os.path.join(LR_down2_path, str(count)+file[-4:]), lr_img)
            count += 1
        else:  # too big downsampling
            lr_img = cv2.resize(lr_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            hr_img = cv2.resize(hr_img, (w, h), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(HR_down2_path, str(count)+file[-4:]), hr_img)
            cv2.imwrite(os.path.join(LR_down2_path, str(count)+file[-4:]), lr_img)
            count += 1
    print("end data cleaning")

# find the most closest images in test
    print("start find closest")
    test_LR_path = "./datasets/test_data/LR"
    save_HR_path = "./datasets/same/HR"
    save_LR_path = "./datasets/same/LR"


    if not os.path.exists("./datasets/same"):
        os.mkdir("./datasets/same")

    if not os.path.exists(save_HR_path):
        os.mkdir(save_HR_path)

    if not os.path.exists(save_LR_path):
        os.mkdir(save_LR_path)

    test_list = os.listdir(test_LR_path)
    train_list = os.listdir(LR_down2_path)
    for file_test in test_list:
        num = int(file_test[:-4])

        print(file_test)
        test_lr = cv2.imread(os.path.join(test_LR_path, file_test), -1)
        h, w = test_lr.shape[0:2]
        test_lr_ = cv2.resize(test_lr, (w // 10, h // 10))
        for file_train in train_list:
            train_lr = cv2.imread(os.path.join(LR_down2_path, file_train), -1)
            train_lr_ = cv2.resize(train_lr, (w // 10, h // 10))
            ssim_ = utils.compute_ssim(test_lr_, train_lr_)
            if ssim_ > 0.65:
                print(file_test, ssim_)
                img_hr = cv2.imread(os.path.join(HR_down2_path, file_train), -1)
                cv2.imwrite(os.path.join(save_HR_path, file_train), img_hr)
                cv2.imwrite(os.path.join(save_LR_path, file_train), test_lr)
                continue
    print("end find closest")


# Augmenting data sets with duplicate data
    print("start augmentation")
    imglist = os.listdir(save_HR_path)
    for i in range(3):
        for file in imglist:
            hr = cv2.imread(os.path.join(save_HR_path, file), -1)
            lr = cv2.imread(os.path.join(save_LR_path, file), -1)
            file_name_tosave = str(count) + file[-4:]
            cv2.imwrite(os.path.join(HR_down2_path, file_name_tosave), hr)
            cv2.imwrite(os.path.join(LR_down2_path, file_name_tosave), lr)
            count += 1

    print("end augmentation")
    return count
