import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import random

import torch
torch.autograd.set_detect_anomaly(True)
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
from torch.nn import DataParallel

from tqdm import tqdm
from models.defineHourglass_512_gray_skip import HourglassNet
from dataset import RelightDataset
from utils.utils_plot import performance_display

import warnings
warnings.filterwarnings("ignore")

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='PyTorch Relight Model Training'
    )

    parser.add_argument('--model-name', type=str, default='HourglassNet512', help='model name')

    # dataset
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')

    parser.add_argument('--output_path', default="output", type=str, help='output path')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    
    return parser.parse_args()

def val(model, data_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (img_input, light_input, img_target, light_target) in enumerate(data_loader):
            img_input = img_input.to(device)
            light_input = light_input.to(device)
            img_target = img_target.to(device)
            light_target = light_target.to(device)

            img_pred, light_pred = model(img_input, light_target, 0)
            img_loss = criterion(img_pred, img_target)
            light_loss = criterion(light_pred, light_input)
            loss = img_loss + light_loss
            test_loss += loss

    test_loss /= len(data_loader)
    print('Test Loss: %.4f' % (test_loss))
    return test_loss

def train(model, data_loader, optimizer, criterion, device, args):
    model.train()
    train_loss = 0
    pbar = tqdm(data_loader)
    for data in pbar:
        img_input, light_input, img_target, light_target  = data
        img_input = img_input.to(device)
        light_input = light_input.to(device)
        img_target = img_target.to(device)
        light_target = light_target.to(device)

        optimizer.zero_grad()
        img_pred, light_pred = model(img_input, light_target, 0)
        img_loss = criterion(img_pred, img_target)
        light_loss = criterion(light_pred, light_input)
        loss = img_loss + light_loss
        pbar.set_postfix_str("loss: {:.4f}, img loss: {:.4f}, light loss: {:.4f}".format(loss.item(), img_loss.item(), light_loss.item()))

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(data_loader)
    print('Train Loss: %.4f' % (train_loss))
    return train_loss

def main(args):

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = RelightDataset('data/train.lst', 'data/DPR_dataset', transform)
    val_data = RelightDataset('data/val.lst', 'data/DPR_dataset', transform)
    test_data = RelightDataset('data/test.lst', 'data/DPR_dataset', transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # ????????????
    model = HourglassNet()
    model = DataParallel(model)
    model = model.to(args.device)
    # print(model)

    optimizor = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizor, T_max=args.epochs, eta_min=1e-6)
    criterion = torch.nn.MSELoss().to(args.device)

    train_losses = []
    val_losses = []
    best_loss = 1e10
    bad_num = 0
    # ????????????
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizor, criterion, args.device, args)
        val_loss = val(model, val_loader, criterion, args.device)
        print('Epoch: %d, Train Loss: %.4f, Val Loss: %.4f' % (epoch, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if args.save_model:
            torch.save(model.state_dict(), '%s/%s_model_%d.pth' % (args.output_path, args.model_name, epoch))
        
        if best_loss > val_loss:
            bad_num = 0
            print('Saving best model...')
            torch.save(model.state_dict(), '%s/%s_best_model.pth' % (args.output_path, args.model_name))
            best_loss = val_loss
        else:
            bad_num += 1

        if bad_num > 10:
            break

        lr_sheduler.step()

    loss_plot = {}
    loss_plot['train_loss'] = train_losses
    loss_plot['val_loss'] = val_losses

    performance_display(loss_plot, 'Loss', args.output_path)

if __name__ == "__main__":
    seed_torch()
    args = parse_arguments()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    main(args)

    # nohup python -u train.py > hourglass-512.log 2>&1 &