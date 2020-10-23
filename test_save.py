import torch
import argparse
from network import SRNDeblurNet
from data_new import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_model, set_requires_grad
from time import time
import os
from skimage.io import imsave

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

input_list = "./testimg/input.txt"
output_dir = "./output"
batch_size = 1
if __name__ == '__main__':

    img_list = open(input_list, 'r').read().strip().split('\n')
    dataset = TestDataset(img_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1,
                            pin_memory=True)

    net = SRNDeblurNet().cuda()
    set_requires_grad(net, False)
    last_epoch = load_model(net, "./save", epoch=1909)

    psnr_list = []
    output_list = []

    tt = time()

    img_num = 0
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for k in batch:
            if 'img' in k:
                batch[k] = batch[k].cuda()
                batch[k].requires_grad = False

        y, _, _ = net(batch['img256'], batch['img128'], batch['img64'])

        y.detach_()
        y = ((y.clamp(-1, 1) + 1.0) / 2.0 * 255.999).byte()
        y = y.permute(0, 2, 3, 1).cpu().numpy()  # NHWC


        for img in y:
            imsave("./output/" + str(img_num) + ".png", img)
            img_num += 1
