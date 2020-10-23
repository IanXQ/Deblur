import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
from torchvision import transforms
from PIL import Image
from kacn import KACN
import time
from data.gopro import GoProDataSet

from datetime import datetime

device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')

'''
计算psnr的函数和计算方法 skimage.measure
skimage.measure.compare_psnr()
skimage.measure.compare_ssim()
transforms.toPILim
'''


def train(net, train_iter, test_iter, num_epochs, batch_size, optimizer, device):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.L1Loss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            n += y.shape[0]
            batch_count += 1
            print("loss is ", l)
        test_psnr = evaluate_accuracy(test_iter, net, device)
        print("test psnr is %.2f" % test_psnr)
        print("epoch %d, loss %.4f" % (epoch + 1, train_l_sum / n))


def evaluate_accuracy(test_iter, test_net, device):
    print("psnr test start!")
    trans = transforms.ToPILImage()
    transtotensor = transforms.ToTensor()
    psnr_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            test_net.eval()
            y_hat = test_net(X)
            test_net.train()
            y_hat = y_hat.cpu()
            for i in range(y_hat.shape[0]):
                target = trans((y[i] + 1) / 2)
                output = trans((y_hat[i] + 1) / 2)
                psnr_sum += peak_signal_noise_ratio(np.array(target), np.array(output), data_range=255)
            n += y.shape[0]
            break
        test_img = Image.open(r"/home/jihuazhu/xuqipeng/SRN-Pytorch/checkpoint/img/test.png")
        img = trans((net((transtotensor(test_img) * 2 - 1).unsqueeze(0).to(device)).cpu()[0] + 1) / 2)
        img.save(r"/home/jihuazhu/xuqipeng/SRN-Pytorch/checkpoint/img/%s_psnr_%.2f.png" % (
            str(datetime.now()), psnr_sum / n))
        torch.save(test_net.state_dict(), r"/home/jihuazhu/xuqipeng/SRN-Pytorch/checkpoint/model/%s_psnr_%.2f.pkl" % (
            str(datetime.now()), psnr_sum / n))
    print("psnr test over!")
    return psnr_sum / n


batch_size = 3
train_loader = DataLoader(dataset=GoProDataSet(), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=GoProDataSet(train=False), batch_size=1)
net = KACN(ngroups=3, nblocks=8, planes=32)
net.load_state_dict(torch.load('para.pkl'))
optimizer = torch.optim.Adam(net.parameters())
train(net, train_iter=train_loader, test_iter=test_loader, num_epochs=500, optimizer=optimizer, device=device,
      batch_size=batch_size)
