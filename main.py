import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim, sigmoid, zeros_like
from tqdm import tqdm
from imageio import imwrite
from eval import eval_net

import torch.nn.functional as F
import LPUIF

from pytorch_msssim import ssim

from utils.dataset import BasicDataset, TestDataset
from torch.utils.data import DataLoader, random_split

dir_img = '/home/s1u1/dataset/roadscene/ir_128/'
dir_mask = '/home/s1u1/dataset/roadscene/vi_128/'


dir_checkpoint = 'checkpoints/'
showpathimg = 'epoch_fuseimg_show/img/'
showpathvis = 'epoch_fuseimg_show/vis/'
showpathinf = 'epoch_fuseimg_show/inf/'
showpathadd = 'epoch_fuseimg_show/add/'
showpathxo = 'epoch_fuseimg_show/xo/'
showpathyo = 'epoch_fuseimg_show/yo/'
showpathfxo = 'epoch_fuseimg_show/fxo/'
showpathfyo = 'epoch_fuseimg_show/fyo/'

loss_mse = nn.MSELoss(reduction='mean').cuda()


def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads



def imgshow(img, showpath, index):
    img = img[1,:,:,:]
    img_final = img.detach().cpu().numpy()
    img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
    img = img_final.transpose(1,2,0)
    img = img.astype('uint8')
    if img.shape[2] == 1:
        img = img.reshape([img.shape[0], img.shape[1]])
    indexd = format(index, '05d')
    file_name = str(indexd) + '.png'
    path_out = showpath + file_name          
    imwrite(path_out, img)
    return img


def caculateloss_v3(fxout, fyout, img, imgs, true_masks):
    m1 = features_grad(fxout)
    m1abs = torch.abs(m1)
    m2 = features_grad(fyout)
    m2abs = torch.abs(m2)
    mone = torch.ones_like(m1)
    mzero = torch.zeros_like(m1)
    mhaf = torch.zeros_like(m1)*0.5
    mask1 = torch.where(m1abs>m2abs,mone,mzero)
    mask1 = torch.where(m1abs==m2abs,mhaf,mask1)
    mask2 = 1-mask1
    weight1 = mask1

    weight2 = mask2

    w1 = img*weight1
    imgs = imgs*weight1
    w2 = img*weight2
    true_masks = true_masks*weight2
    loss_1 = (1 - ssim(w1, imgs)) + (1 - ssim(w2, true_masks))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(w1, imgs) + loss_mse(w2, true_masks)
    loss_2 = torch.mean(loss_2)

    x1 = torch.mean(fxout)
    x2 = torch.mean(fyout)

    '''yuanlai'''
    bb1 = torch.sqrt(torch.sigmoid(fxout-x1))
    bb2 = torch.sqrt(torch.sigmoid(fyout-x2))
    b1 = bb1/(bb1+bb2)

    b2 = bb2/(bb1+bb2)
    b1 = torch.sigmoid(b1-torch.mean(b1))
    b2 = torch.sigmoid(b2-torch.mean(b2))

    bw1 = img*b1
    bimgs = imgs*b1
    bw2 = img*b2
    btrue_masks = true_masks*b2
    loss_3 = loss_mse(bw1, bimgs) + loss_mse(bw2, btrue_masks)
    loss_3 = torch.mean(loss_3)

    alph = 100

    loss = loss_1 + alph * loss_2 + 0.4*loss_3

    # loss = loss_1 + 20 * loss_2 + 0.4*loss_3
    return loss, weight1, weight2

def base(fxout, fyout, img, imgs, true_masks):
    weight1 = 1
    weight2 = 1
    w1 = img*weight1
    imgs = imgs*weight1
    w2 = img*weight2
    true_masks = true_masks*weight2
    loss_1 = (1 - ssim(w1, imgs)) + (1 - ssim(w2, true_masks))
    loss_1 = torch.mean(loss_1)

    loss_2 = loss_mse(w1, imgs) + loss_mse(w2, true_masks)
    loss_2 = torch.mean(loss_2)
    b1 = 1
    b2 = 1
    bw1 = img*b1
    bimgs = imgs*b1
    bw2 = img*b2
    btrue_masks = true_masks*b2
    loss_3 = loss_mse(bw1, bimgs) + loss_mse(bw2, btrue_masks)
    loss_3 = torch.mean(loss_3)

    loss = loss_1 + 20 * loss_2 + 0.4*loss_3
    return loss, imgs, true_masks

def train_net(net, 
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.25):
    ph = 1
    al = 0 #低于此全部置0
    c = 3500

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    train, val1 = random_split(dataset, [n_train, n_val])
    n_val9 = int(n_val*0.5)
    n_val1 = n_val-n_val9
    val2, val = random_split(val1, [n_val9, n_val1])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    # writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    index = 1
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-7, momentum=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(epochs):
        net.train()

        epoch_loss1 = 0
        epoch_loss2 = 0
        epoch_loss3 = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs2 = batch['image2']
                true_masks2 = batch['mask2']
                imgs3 = batch['image3']
                true_masks3 = batch['mask3']
                imgs4 = batch['image4']
                true_masks4 = batch['mask4']
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()
                    imgs2 = imgs2.cuda()
                    true_masks2 = true_masks2.cuda()
                    imgs3 = imgs3.cuda()
                    true_masks3 = true_masks3.cuda()
                    imgs4 = imgs4.cuda()
                    true_masks4 = true_masks4.cuda()
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                imgs2 = imgs2.to(device=device, dtype=torch.float32)
                true_masks2 = true_masks2.to(device=device, dtype=torch.float32)
                imgs3 = imgs3.to(device=device, dtype=torch.float32)
                true_masks3 = true_masks3.to(device=device, dtype=torch.float32)
                imgs4 = imgs4.to(device=device, dtype=torch.float32)
                true_masks4 = true_masks4.to(device=device, dtype=torch.float32)

                img, vis, inf, xout, yout, fxout, fyout = net(imgs, true_masks)

############像素级loss
                # loss1, weight1, weight2 = caculateloss(fxout, fyout, img, imgs, true_masks)
                loss1, weight1, weight2 = caculateloss_v3(fxout, fyout, img, imgs, true_masks)
                # loss1, weight1, weight2 = base(fxout, fyout, img, imgs, true_masks)
                lossall = loss1# + loss3

                pbar.set_postfix(**{'loss1 (batch)': loss1.item()})#,'loss3 (batch)': loss3.item(),'loss3 (batch)': loss3.item()})
                optimizer.zero_grad()
                # loss2.backward(retain_graph=True)
                # loss1.backward()
                # loss3.backward()
                lossall.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                ######
                if global_step == ((n_train // batch_size)*index):
                    g = imgshow(img,showpathimg,index)
                    print(optimizer.state_dict()['param_groups'][0]['lr'])
#################
                    index += 1  
                #####
                if global_step % (n_train // (1  * batch_size)) == 0:
                # if global_step == 5:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        # value.requires_grad = False
                        # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    # writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--test', dest='test', type=str, default=True,
                        help='If test images turn True, train images turn False')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    pthf = 'checkpoints/CP_epoch25.pth'


    ir = '/home/s1u1/dataset/M3FD/M3FD_Detection/ir_1ch/'
    vi = '/home/s1u1/dataset/M3FD/M3FD_Detection/vi_1ch/'

    path = '/home/s1u1/code/3line/outputsall/M3FD_detection/'
    pathadd = './outputsadd/'


    dataset = TestDataset(ir, vi)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    net = LPUIF.wtNet(n_channels=1, n_classes=1, bilinear=True, pthfile=pthf)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    index = 1
    Time = 0
    if args.test:
        for im in test_loader:
            
            ir = im['image']
            vi = im['mask']
            if torch.cuda.is_available():
                ir = ir.cuda()
                vi = vi.cuda()
            # Net = Wdenet.wtNet(1, 1)
            # Net = unet_model.UNet(1,1)
            Net = LPUIF.wtNet(1, 1, pthfile=pthf)
            Net = Net.cuda()

            start = time.time()

            img,_,_,_,_,_,_ = Net(vi, ir)
            # img = Net(vi, ir)
            img_final = img.detach().cpu().numpy() 
            img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
            # img = img.clamp(0, 255).data[0].numpy()
            img = img_final.transpose(1, 2, 3, 0)
            img = img.squeeze(0).astype('uint8')
            if img.shape[2] == 1:
                img = img.reshape([img.shape[0], img.shape[1]])
            end = time.time()
            indexd = format(index, '05d')
            file_name = str(indexd) + '.png'
            path_out = path + file_name
            # index += 1            
            imwrite(path_out, img)
            Time += end-start
            print(index)
            print(end-start)
            index += 1  
        average_time = Time/(len(test_loader))  
        print(average_time) 

    else:
        try:
            train_net(net=net,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    device=device,
                    img_scale=args.scale,
                    val_percent=args.val / 100)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
