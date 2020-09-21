import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import model
from data import DIV2K
from utils import utils as utils
import skimage.color as sc
import random
from collections import OrderedDict
from data import datacleaning_augmentation as datapro
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from data.mydataloader import Dataset
from data import mydataloader

# import cv2
# LR_down2_path = "./datasets/train_data/HR_Down2"
# path_list = os.listdir(LR_down2_path)
# for file in path_list:
#     lr_img = cv2.imread(os.path.join(LR_down2_path, file), -1)
#     h, w = lr_img.shape[:2]
#     if h<540 or w<540:
#         print(file,h,w)
# print("end")

# Training settings
parser = argparse.ArgumentParser(description="IMDN")
parser.add_argument("--choose_net", type=str, default="model",#16
                    help="model")
parser.add_argument("--batch_size", type=int, default=48,#16
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=0, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=0,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=2128,#2302
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=400)#1000
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=528,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="./pretrained-model/model.pth", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.jpg')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--psnr_ssim_max", type=float, default=1.0)

args = parser.parse_args()
# print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
# print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> data preprocessing")
args.n_train=datapro.dataprocessing()
print("num of images {}".format(args.n_train))
print("===> Loading datasets")
trainset = DIV2K.div2k(args)
testset =Dataset("./datasets/same/HR", "./datasets/same/LR")
# train_dataset = mydataloader.TrainDataset(args.train_file, patch_size=args.patch_size, scale=args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                  shuffle=False)

print("===> Building models")
args.is_train = True

modeltrain = model.model(upscale=args.scale)

# model = baseline.model(upscale=args.scale)

l2_criterion = nn.MSELoss()

print("===> Setting GPU")
if cuda:
    modeltrain = modeltrain.to(device)
    criterion2 = l2_criterion.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        # model_dict = modeltrain.state_dict()
        # pretrained_dict = torch.load(args.pretrained)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # modeltrain.load_state_dict(model_dict)

        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = modeltrain.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        modeltrain.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")

optimizer = optim.Adam(modeltrain.parameters(), lr=args.lr,betas=(0.9, 0.999), eps=1e-8)


def train(epoch):
    modeltrain.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        # print(torch.min(lr_tensor),torch.max(lr_tensor),torch.min(hr_tensor),torch.max(hr_tensor))
        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = modeltrain(lr_tensor)
        loss_l2 = l2_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l2

        loss_sr.backward()
        optimizer.step()
        # print("epoch: {} loss: {}".format(epoch,loss_sr.item()))
        if iteration % 50 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_sr.item()))


def valid():
    modeltrain.eval()

    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        # print(torch.min(lr_tensor), torch.max(lr_tensor), torch.min(hr_tensor), torch.max(hr_tensor))
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = modeltrain(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)

    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))
    # if args.psnr_ssim_max <= ((avg_psnr/ len(testing_data_loader)-30.8804) * 2 + (avg_ssim/ len(testing_data_loader)-0.8782) * 4):
    #     args.psnr_ssim_max = (avg_psnr/ len(testing_data_loader)-30.8804) * 2 + (avg_ssim/ len(testing_data_loader)-0.8782) * 4
    #     print("score:"+str(args.psnr_ssim_max))
    #     return 1
    if args.psnr_ssim_max <= ((avg_psnr/ len(testing_data_loader)-30.6561) * 2 + (avg_ssim/ len(testing_data_loader)-0.8819) * 4):
        args.psnr_ssim_max = (avg_psnr/ len(testing_data_loader)-30.6561) * 2 + (avg_ssim/ len(testing_data_loader)-0.8819) * 4
        print("score:"+str(args.psnr_ssim_max))
        return 1
    return 0


def save_checkpoint(epoch):
    model_folder = "./pretrained-model/"
    model_out_path = model_folder + "model.pth"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(modeltrain.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(modeltrain)
for epoch in range(args.start_epoch, args.nEpochs + 1):
    train(epoch)
    if valid():
        save_checkpoint(epoch)

# best score: psnr=8.619773372258521，ssim=0.29088628652944015,time=0.09999999999999999~0.05217391304347826