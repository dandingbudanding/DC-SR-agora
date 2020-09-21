from model import model
from utils import utils
from torch.utils.data import DataLoader
import torch
import skimage.color as sc
from data.mydataloader import Dataset
import time
import cv2
import os

cuda = 1
device = torch.device('cuda' if cuda else 'cpu')

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

# load model
model = model.model().to(device)
model_dict = utils.load_state_dict("./pretrained-model/model.pth")
model.load_state_dict(model_dict, strict=True)
# load model

testset =Dataset("./datasets/same/HR", "./datasets/same/LR")
testing_data_loader = DataLoader(dataset=testset, num_workers=0, batch_size=1,
                                 shuffle=False)

model.eval()

avg_psnr, avg_ssim = 0, 0
start = time.time()


for _,batch in enumerate(testing_data_loader):
    lr_tensor, hr_tensor = batch[0], batch[1]
    if 1:
        lr_tensor = lr_tensor.to(device)
        hr_tensor = hr_tensor.to(device)

    with torch.no_grad():
        pre = model(lr_tensor)

    sr_img = utils.tensor2np(pre.detach()[0])
    gt_img = utils.tensor2np(hr_tensor.detach()[0])
    crop_size = 2
    cropped_sr_img = utils.shave(sr_img, crop_size)
    cropped_gt_img = utils.shave(gt_img, crop_size)

    im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
    im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])

    img_save=sr_img.transpose(2,0,1)

    cv2.imwrite(os.path.join("./results/",str(_)+".png"),img_save)

    avg_psnr += utils.compute_psnr(im_pre, im_label)
    avg_ssim += utils.compute_ssim(im_pre, im_label)

end = time.time()-start
print(end)
print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))




# model3  40.96315050125122  41.17648005485535

# baseline 41.0912070274353  41.6964111328125