import torch
import os
import numpy as np

class DataSet(torch.utils.data.Dataset):

    def __init__(self, voxels, hard_element_stiffness, hard_element_force, soft_element_stiffness, soft_element_force, C_hard, C_soft, jacobi, voxel_length, case,device):
        self.voxels = torch.tensor(voxels).float().to(device)
        self.hard_element_stiffness = torch.tensor(hard_element_stiffness).float().to(device)
        self.hard_element_force = torch.tensor(hard_element_force).float().to(device)
        self.soft_element_stiffness = torch.tensor(soft_element_stiffness).float().to(device)
        self.soft_element_force = torch.tensor(soft_element_force).float().to(device)
        self.C_hard = torch.tensor(C_hard).float().to(device)
        self.C_soft = torch.tensor(C_soft).float().to(device)
        self.jacobi = torch.tensor(jacobi).float().to(device)
        self.voxel_length = voxel_length
        self.case = case

        self.voxel_length = voxel_length

    def __getitem__(self, index):
        voxel = self.voxels[index // self.case]
        hard_stiffness = self.hard_element_stiffness[index]
        hard_force = self.hard_element_force[index]
        soft_stiffness = self.soft_element_stiffness[index]
        soft_force = self.soft_element_force[index]
        c_hard = self.C_hard[index]
        c_soft = self.C_soft[index]
        jacobi_ = self.jacobi[index]
        input = (c_hard.reshape(36, 1) @ voxel.reshape(1, -1) + c_soft.reshape(36, 1) @ (1 - voxel).reshape(1, -1)).reshape(36, self.voxel_length,
                                                                                          self.voxel_length, self.voxel_length)

        return input, voxel, hard_stiffness, hard_force, soft_stiffness, soft_force, c_hard, c_soft, jacobi_

    def __len__(self):
        return len(self.jacobi)


def load_dataset(base_path, voxel_length, case,train_ratio,device):

    data=np.load(os.path.join(base_path,'data.npz'))

    voxels=data['voxel']
    hard_element_stiffness_train=data['Ke_hard']
    hard_element_force_train=data['Fe_hard']
    soft_element_stiffness_train=data['Ke_soft']
    soft_element_force_train=data['Fe_soft']
    C_hard=data['C_hard']
    C_soft=data['C_soft']
    jacobi=data['J']
    dataset = DataSet(voxels,  hard_element_stiffness_train, hard_element_force_train, soft_element_stiffness_train, soft_element_force_train, C_hard, C_soft, jacobi, voxel_length, case,device)
    
    # make 80% for train_dateset and 20% for test_dataet 
    # print(len(dataset))
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset