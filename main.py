## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float,dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)


if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))


if mode == 'train':
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    # transforms.ElasticTransform(),
    transforms.ToTensor()])

    dataset_train = Custom_Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = Custom_Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([transforms.ToTensor()])

    dataset_test = Custom_Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)


    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

model = UNet().to(device)

criterion = nn.CrossEntropyLoss().to(device)

optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)


fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

# Threshold = 0.5
fn_class = lambda x: 1.0 * (x > 0.5)


writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


st_epoch = 0

# Train mode
if mode == 'train':
    if train_continue == "on":
        model, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=model, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        model.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            input = data[0].to(device)
            label = data[1].to(device)

            output = model(input)

            # backward pass
            optim.zero_grad()

            loss = criterion(output, label.squeeze())
            loss.backward()

            optim.step()

            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            output_img = torch.argmax(fn_class(output),dim=1).unsqueeze(1) * 255

            writer_train.add_images('label', label, num_batch_train * (epoch - 1) + batch)
            writer_train.add_images('input', input, num_batch_train * (epoch - 1) + batch)
            writer_train.add_images('output',output_img , num_batch_train * (epoch - 1) + batch)

        writer_train.add_scalar('Epoch_loss', np.mean(loss_arr), epoch)

        print(f'---------Epoch{epoch} Training Finish---------')

        with torch.no_grad():
            model.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                input = data[0].to(device)
                label = data[1].to(device)
                
                output = model(input)

                loss = criterion(output, label.squeeze())

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))


                output_img = torch.argmax(fn_class(output),dim=1).unsqueeze(1) * 255

                writer_val.add_images('label', label, num_batch_val * (epoch - 1) + batch)
                writer_val.add_images('input', input, num_batch_val * (epoch - 1) + batch)
                writer_val.add_images('output', output_img, num_batch_val * (epoch - 1) + batch)

        writer_val.add_scalar('Epoch_loss', np.mean(loss_arr), epoch)

        
        save(ckpt_dir=ckpt_dir, net=model, optim=optim, epoch=epoch)
        print(f'---------Epoch{epoch} Valid Finish---------')

    writer_train.close()
    writer_val.close()

# Test mode
else:
    model, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=model, optim=optim)

    with torch.no_grad():
        model.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):

            input = data[0].to(device)
            label = data[1].to(device)

            output = model(input)

            loss = criterion(output, label.squeeze())

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))
            
            output_img = torch.argmax(fn_class(output),dim=1).unsqueeze(1).detach().cpu().numpy()
            input = input.detach().cpu().numpy()
            label= label.detach().cpu().numpy()

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output_img[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output_img[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))

