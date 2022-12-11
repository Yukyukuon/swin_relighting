import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
import argparse

from tqdm import tqdm
import models
from dataset import RelightDataset

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

    parser.add_argument('--model-name', type=str, default='resnet50', help='model name')

    # dataset
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 100)')

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')


    parser.add_argument('--output_path', default="output", type=str, help='output path')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    
    return parser.parse_args()

def test(model, data_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    item_num = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()

            item_num += labels.size(0)
    test_loss /= len(data_loader)
    print('Test Loss: %.4f' % (test_loss))
    print('Test Accuracy: %.4f' % (correct))
    return test_loss

def train(model, data_loader, optimizer, criterion, device, args):
    model.train()
    train_loss = 0
    correct = 0
    item_num = 0
    pbar = tqdm(data_loader)
    for data in pbar:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        item_num += labels.size(0)
    train_loss /= len(data_loader)
    print('Train Loss: %.4f' % (train_loss))
    return train_loss

def main(args):

    train_data = RelightDataset('train')
    test_data = RelightDataset('test')
    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    # 定义模型
    model = getattr(models, args.model_name)(num_classes=len(train_data.classes)).to(args.device)
    print(model)

    optimizor = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss().to(args.device)

    train_losses = [], []
    test_losses = [], []
    best_acc = 0
    bad_num = 0
    # 开始训练
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_load, optimizor, criterion, args.device, args)
        test_loss, test_acc = test(model, test_load, criterion, args.device)
        print('Epoch: %d, Train Loss: %.4f, Test Loss: %.4f' % (epoch, train_loss, train_acc, test_loss, test_acc))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if args.save_model:
            torch.save(model.state_dict(), '%s/%s_model_%d.pth' % (args.output_path,args.model_name, epoch))
        if best_acc < train_acc:
            bad_num = 0
            print('Saving best model...')
            torch.save(model.state_dict(), '{}_best_model.pth'.format(args.model_name))
            best_acc = train_acc
        else:
            bad_num += 1

        if bad_num > 10:
            break

    loss_plot = {}
    loss_plot['train_loss'] = train_losses
    loss_plot['test_loss'] = test_losses


if __name__ == "__main__":
    seed_torch()
    args = parse_arguments()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    main(args)