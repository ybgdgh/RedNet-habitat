import argparse
import os
import time
import torch

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn
import numpy as np

from tensorboardX import SummaryWriter
from habitat_sim.utils.common import d3_40_colors_rgb


import RedNet_model
import RedNet_data
from PIL import Image
import matplotlib.pyplot as plt
from RedNet_model import load_rednet, save_ckpt
from utils import print_log
from torch.optim.lr_scheduler import LambdaLR
import utils


import cv2

parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=14, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=10, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480

def display_sample(rgb_obs, semantic_obs, depth_obs, count_steps):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")

    # arr = [rgb_img, semantic_img, depth_img]

    # titles = ['rgb', 'semantic', 'depth']
    # plt.figure(figsize=(12 ,8))
    # for i, data in enumerate(arr):
    #     ax = plt.subplot(1, 3, i+1)
    #     ax.axis('off')
    #     ax.set_title(titles[i])
    #     plt.imshow(data)
    # plt.show()

    rgb_img = cv2.cvtColor(np.asarray(rgb_img),cv2.COLOR_RGB2BGR)
    sem_img = cv2.cvtColor(np.asarray(semantic_img),cv2.COLOR_RGB2BGR)

    fn = 'result_target/Vis-rgb-{}.png'.format(count_steps)
    cv2.imwrite(fn, rgb_img)
    fn = 'result_target/Vis-sem-{}.png'.format(count_steps)
    cv2.imwrite(fn, sem_img)

    # cv2.imshow("RGB", rgb_img)
    # # cv2.imshow("depth_img", depth_img)
    # cv2.imshow("Sematic", sem_img)

def train():

    train_data = RedNet_data.SUNRGBD(transform=transforms.Compose([RedNet_data.scaleNorm(),
                                                                   RedNet_data.RandomScale((1.0, 1.4)),
                                                                   RedNet_data.RandomHSV((0.9, 1.1),
                                                                                         (0.9, 1.1),
                                                                                         (25, 25)),
                                                                   RedNet_data.RandomCrop(image_h, image_w),
                                                                   RedNet_data.RandomFlip(),
                                                                   RedNet_data.ToTensor()]),
                                     phase_train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    # dataitr = iter(train_loader)

    num_train = len(train_data)
    print("num_train: ", num_train)
    # print("train_loader: ", train_loader)

    model = load_rednet(
        device, ckpt='model/rednet_semmap_mp3d_40.pth', resize=True, # since we train on half-vision
    )

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    CEL_weighted = utils.CrossEntropyLoss2d()
    model.train()
    model.to(device)
    CEL_weighted.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    global_step = 0

    lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    writer = SummaryWriter(args.summary_dir)

    for epoch in range(args.epochs):

        local_count = 0
        last_count = 0
        end_time = time.time()
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                      local_count, num_train)

        for batch_idx, sample in enumerate(train_loader):

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
            # print("image: ", image.shape) # torch.Size([2, 3, 480, 640])
            # print("depth: ", depth.shape) # torch.Size([2, 1, 480, 640])
            # print("target_scales: ", target_scales[0].shape) # torch.Size([2, 480, 640])
            # display_sample(image[0].cpu().detach().numpy(), target_scales[0][0].cpu().detach().numpy(), depth[0][0].cpu().detach().numpy(), batch_idx)
            # print("target_scales: ",len(target_scales))
            optimizer.zero_grad()
            pred_scales = model(image, depth, train=True)
            # print("pred_scales: ", len(pred_scales))
            loss = CEL_weighted(pred_scales, target_scales)
            # print("loss: ", loss)
            loss.backward()
            optimizer.step()
            local_count += image.data.shape[0]
            global_step += 1
            if global_step % args.print_freq == 0 or global_step == 1:

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_log(global_step, epoch, local_count, count_inter,
                        num_train, loss, time_inter)
                end_time = time.time()

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
                grid_image = make_grid(image[1:].permute(0, 3, 1, 2).clone().cpu().data, 3, normalize=True)
                writer.add_image('image', grid_image, global_step)
                grid_image = make_grid(depth[1:].permute(0, 3, 1, 2).clone().cpu().data, 3, normalize=True)
                writer.add_image('depth', grid_image, global_step)
                grid_image = make_grid(utils.color_label(torch.max(pred_scales[0][:3], 1)[1] + 1), 3, normalize=False,
                                    range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)
                writer.add_scalar('CrossEntropyLoss', loss.data, global_step=global_step)
                writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)
                last_count = local_count

        scheduler.step(epoch)

    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
              0, num_train)

    print("Training completed ")

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    train()
