import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RelightDataset(Dataset):
    def __init__(self, list_file, data_dir, transform=None):
        self.img_inputs = []
        self.light_inputs = []
        self.img_targets = []
        self.light_targets = []

        with open(list_file, 'r') as f:
            for line in f:
                img_name, img_input_name, img_target_name = line.strip().split(' ')
                img_input_path = os.path.join(data_dir, img_name, img_input_name)
                self.img_inputs.append(img_input_path)
                light_input_name = img_input_name.split('.')[0].split('_')[0] + '_light_' + img_input_name.split('.')[0].split('_')[1] + '.txt'
                light_input_path = os.path.join(data_dir, img_name, light_input_name)
                self.light_inputs.append(light_input_path)
                img_target_path = os.path.join(data_dir, img_name, img_target_name)
                self.img_targets.append(img_target_path)
                light_target_name = img_target_name.split('.')[0].split('_')[0] + '_light_' + img_target_name.split('.')[0].split('_')[1] + '.txt'
                light_target_path = os.path.join(data_dir, img_name, light_target_name)
                self.light_targets.append(light_target_path)

        self.transform = transform


    def __len__(self):
        return len(self.img_inputs)

    def __getitem__(self, index):
        img_input_path = self.img_inputs[index]
        img_input = Image.open(img_input_path).convert('RGB')
        img_input = self.transform(img_input)[:1,:,:]

        light_input_path = self.light_inputs[index]
        light_input = np.loadtxt(light_input_path) * 0.7
        light_input = np.reshape(light_input, (9,1,1))
        light_input = torch.from_numpy(light_input).float()

        img_target_path = self.img_targets[index]
        img_target = Image.open(img_target_path).convert('RGB')
        img_target = self.transform(img_target)[:1,:,:]

        light_target_path = self.light_targets[index]
        light_target = np.loadtxt(light_target_path) * 0.7
        light_target = np.reshape(light_target, (9,1,1))
        light_target = torch.from_numpy(light_target).float()

        return img_input, light_input, img_target, light_target

if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = RelightDataset('data/val.lst', 'data/DPR_dataset', transform)
    img_input, light_input, img_target, light_target = dataset[0]
    print(img_input.shape, light_input.shape, img_target.shape, light_target.shape)