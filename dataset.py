import torch
from torch.utils.data import Dataset

class RelightDataset(Dataset):
    def __init__(self, img_inputs, light_inputs, img_targets, light_targets):
        self.img_inputs = img_inputs
        self.light_inputs = light_inputs
        self.img_targets = img_targets
        self.light_targets = light_targets


    def __len__(self):
        return len(self.img_inputs)

    def __getitem__(self, index):
        img_input = self.img_inputs[index]
        light_input = self.light_inputs[index]
        img_target = self.img_targets[index]
        light_target = self.light_targets[index]

        return img_input, light_input, img_target, light_target