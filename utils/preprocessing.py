from ctypes import sizeof
from distutils.log import debug
import pandas as pd
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class HARdataset():
    """Build dataset from motion sensor data."""

    def __init__(self, root):
        self.df = pd.read_csv(root, low_memory=False)

        self.parts = ["belt", "arm", "dumbbell", "forearm"]
        # 4 x 10
        self.variables = ["roll_{}", "pitch_{}", "yaw_{}", "total_accel_{}",
                          "accel_{}_x", "accel_{}_y", "accel_{}_z", "gyros_{}_x",
                          "gyros_{}_y", "gyros_{}_z"]
        self.var_list, self.labels, self.mean, self.std = self.normalize_data()
        self.length = self.var_list.size()[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.var_list[:, idx]
        step = torch.unsqueeze(step, 0)
        target = self.labels[idx]

        return step, target, idx

    def normalize_data(self):
        """Returns normalized data."""

        var_list, labels = self.build_dataset()
        var_std = var_list.std(dim=1, keepdim=True)
        var_mean = var_list.mean(dim=1, keepdim=True)
        var_list = (var_list - var_mean) / var_std
		# torch.set_printoptions(profile="full")

        return var_list, labels, var_mean, var_std

    def build_dataset(self):
        """Get list of motion sensor variables and labels."""
        # Fazer isso ficar 3x40 (target final)
        var_list = []
		# torch.set_printoptions(profile="full")                                                                                                         
        torch.set_printoptions(profile="full")
        # for times in range(3):
        for part in self.parts:
            for var in self.variables:
                print(var)
                var_list.append(list(self.df[var.format(part)]))
                print(len(var_list))
  
        var_list = torch.tensor(var_list)

        sub_var_list = []
        var_list_3x40 = []
    
        # var_list_140  =  {  T1 & T2 & T3 ,  T1 & T2 & T3 , T1 & T2 & T3 ,  T1 & T2 & T3 .....  }
        # var_list_3x40  =  {  {T1,T2,T3} ,  T1 & T2 & T3 , T1 & T2 & T3 ,  T1 & T2 & T3 .....  }


        # if ver se mesmo name todos os timesteps

        print("\n\n------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>-------------------------------\n\n")

        for line in range(2,len(var_list[0])):
            for item in var_list:
                sub_var_list.append(item[line])
                sub_var_list.append(item[line-1])
                sub_var_list.append(item[line-2])
            var_list_3x40.append(sub_var_list)
            sub_var_list.clear()


		# decidir qual label colocar - rafael
        labels = torch.tensor([ord(char) for char in list(self.df["classe"])])
        labels -= 65

 
            
        # var_list_3x40 = [list(x) for x  in zip(*var_list_3x40)] #

        print(len(labels))
        labels = labels[2:] 
        print(len(labels))

        print("varlist140 Prints:")
        print(len(var_list_3x40))
        print(len(var_list_3x40[0]))
 
        print("varlistdefauly Prints:")
        print(len(var_list))
        print(len(var_list[0]))
 
        var_list_3x40 = torch.tensor(var_list_3x40)


        return var_list_3x40, labels

    def split_ind(self, val_split, shuffle=True):
        """
        Splits the dataset into training and validation datasets.

        Params:
        val_split: float
                split ratio of dataset
        shuffle: boolean
                shuffle indices if true

        Returns:
        train_sampler, val_sampler
        """

        random_seed = 42

        # Create data indices for training and validation splits
        indices = list(range(self.length))
        split = int(np.floor(val_split * self.length))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Create pytorch data samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        return train_sampler, val_sampler
