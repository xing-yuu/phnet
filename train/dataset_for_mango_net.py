import torch
import os
import numpy as np

class DataSet(torch.utils.data.Dataset):

    def __init__(self,voxels,base_material,device):
        self.voxels =voxels
        self.base_material=base_material
        self.device=device

    def __getitem__(self, index):
        voxel = torch.from_numpy(self.voxels[index]).to(self.device).float()
        voxel_basis=torch.cat((voxel.unsqueeze(0),(1.-voxel).unsqueeze(0)),0)
        input= torch.einsum('bi,bjkl->ijkl',self.base_material,voxel_basis)
        return input,voxel

    def __len__(self):
        return self.voxels.shape[0]


def load_dataset(base_path,base_material,train_ratio,device):

    data=np.load(os.path.join(base_path,'data.npz'))
    voxel_nbr=data['voxel_nbr']
    res=data['res']
    voxels=np.unpackbits(data['voxels']).reshape(voxel_nbr,res,res,res)
    
    dataset = DataSet(voxels,base_material,device)
    
    # make 80% for train_dateset and 20% for test_dataet 
    # print(len(dataset))
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"""==================================dataset information=================================
    Number of microstructure: {voxel_nbr}
    Resolution: {res} * {res} * {res}
    Train ratio: {train_ratio}
    """)


    return train_dataset, test_dataset