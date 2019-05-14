import argparse
import collections
import datetime
import imp
import os
import pickle
import time
import lmdb
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from MobileNetV3 import *
from utils.AverageMeter import AverageMeter
from utils.Logger import Logger



parser = argparse.ArgumentParser(description='imagenet_pretrain')
parser.add_argument("--exp", type=str, default="test", help="experiment")

parser.add_argument("--model_mode", type=str,
                    default="SMALL", help="model_mode")
parser.add_argument('--multiplier', type=float,
                    default=1.0, help="(default: 1.0)")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument('--gpu', type=str, default="0,1,2,3", help='choose GPU')
parser.add_argument("--train_data_dir", type=str,
                    default="", help="train_data_dir")
parser.add_argument("--val_data_dir", type=str,
                    default="", help="val_data_dir")



args = parser.parse_args()




basic_configs = collections.OrderedDict()
basic_configs['serial_number'] = args.exp
basic_configs['learning_rate'] = 1e-1
basic_configs['num_epochs'] = 10000
basic_configs["lr_protocol"] = [(105, 1e-1), (125, 1e-2), (135, 1e-3), (145, 1e-4), (155, 1e-5)]
basic_configs["display_step"] = 20
lr_protocol = basic_configs["lr_protocol"]


dataloader_configs = collections.OrderedDict()
dataloader_configs['batch_size'] = args.batch_size
dataloader_configs['num_workers'] = args.num_workers
dataloader_configs['train_data_dir'] = args.train_data_dir
dataloader_configs['val_data_dir'] = args.val_data_dir


print '==> Preparing Data...'
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.ImageFolder(
    root=dataloader_configs['train_data_dir'], transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=dataloader_configs['batch_size'], shuffle=True, num_workers=dataloader_configs['num_workers'])

val_set = torchvision.datasets.ImageFolder(
    root=dataloader_configs['val_data_dir'], transform=transform_val)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=dataloader_configs['batch_size'], num_workers=dataloader_configs['num_workers'])


# 
exp_dir = os.path.join('./experimental_results', args.exp)

exp_log_dir = os.path.join(exp_dir, "log")
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)

exp_visual_dir = os.path.join(exp_dir, "visual")
if not os.path.exists(exp_visual_dir):
    os.makedirs(exp_visual_dir)

exp_ckpt_dir = os.path.join(exp_dir, "checkpoints")
if not os.path.exists(exp_ckpt_dir):
    os.makedirs(exp_ckpt_dir)

now_str = datetime.datetime.now().__str__().replace(' ', '_')
writer_path = os.path.join(exp_visual_dir, now_str)
writer = SummaryWriter(writer_path)

logger_path = os.path.join(exp_log_dir, now_str + ".log")
logger = Logger(logger_path).get_logger()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

logger.info("basic configuration settings: {}".format(basic_configs))



loss_function = nn.CrossEntropyLoss()
training_loss = AverageMeter()


print '==> Building Model...'
net = MobileNetV3()





net = torch.nn.DataParallel(
    net, device_ids=[int(x) for x in args.gpu.split(',')]).cuda()

optimizer = torch.optim.SGD(net.parameters(
), lr=basic_configs['learning_rate'], momentum=0.9, weight_decay=1e-5)


def train_function(epoch):
    training_loss.reset()
    net.train()

    lr = next((lr for (max_epoch, lr) in lr_protocol if max_epoch >
               epoch), lr_protocol[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info("set learning rate to: {}".format(lr))

    for idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)

        batch_loss = loss_function(outputs, targets)

        batch_loss.backward()

        optimizer.step()

        training_loss.update(batch_loss.item())

        if (idx + 1) % basic_configs["display_step"] == 0:
            logger.info(
                "==> Iteration [{}][{}/{}]:".format(epoch + 1, idx + 1, len(train_loader)))
            logger.info("current batch loss: {}".format(
                batch_loss.item()))
            logger.info("average loss: {}".format(
                training_loss.avg))

    writer.add_scalars("loss", {"training_loss": training_loss.avg}, epoch + 1)


if __name__ == '__main__':

    logger.info("training status: ")
    for epoch in range(basic_configs['num_epochs']):
        logger.info("Begin training epoch {}".format(epoch + 1))
        train_function(epoch)

        net_checkpoint_name = args.exp + \
            "_MobileNetV3_epoch_" + str(epoch + 1)
        net_checkpoint_path = os.path.join(exp_ckpt_dir, net_checkpoint_name)
        net_state = {"epoch": epoch + 1,
                     "network": net.module.state_dict()}
        torch.save(net_state, net_checkpoint_path)
