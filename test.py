import torch
import numpy as np
import os
from train.numerical_homogenization import *
from train.network_homogenization import *
from datagen.fem_helper import *
import time



device = torch.device('cuda')
base_path='dataset/tg/20220406'

data=np.load(os.path.join(base_path,'data.npz'))
data_wrong=np.load(os.path.join(base_path,'data_wrong.npz'))

voxels=torch.from_numpy(data['voxel']).double().to(device)
hard_element_stiffness_train=torch.from_numpy(data['Ke_hard']).double().to(device)
hard_element_force_train=torch.from_numpy(data['Fe_hard']).double().to(device)
soft_element_stiffness_train=torch.from_numpy(data['Ke_soft']).double().to(device)
soft_element_force_train=torch.from_numpy(data['Fe_soft']).double().to(device)
C_hard=torch.from_numpy(data['C_hard']).double().to(device)
C_soft=torch.from_numpy(data['C_soft']).double().to(device)
jacobi=torch.from_numpy(data['J']).double().to(device)

C_hard_wrong=torch.from_numpy(data_wrong['C_hard']).double().to(device)

for i in range(600):
    print(torch.dist(C_hard[i],C_hard_wrong[i])/C_hard[i].norm())






E_hard=1
v=0.3

C_hard_0=isotropic_elastic_tensor(E_hard,v).cuda().double()

hex=torch.tensor([[0,0,0],[1.,0,0],[1.,1.,0],[0,1.,0],[0,0,1.],[1.,0,1.],[1.,1.,1.],[0,1.,1.]],dtype=C_hard.dtype,device=C_hard.device)/36

ke,fe=stiffness_force(hex,C_hard_0)

np.savez('stiffness_force_2.npz',Ke=ke.cpu().numpy(),Fe=fe.cpu().numpy())

r=36

num_homo=numerical_homogenization(r,r,r,1.,1.,1.,device)
net_homo=network_homogenization(r,r,r,1.,1.,1.,device)

voxel=torch.ones_like(voxels[0])

U=num_homo.solve_by_torch(voxel,hard_element_stiffness_train[0],hard_element_force_train[0])



idx = torch.ones(24, dtype=torch.bool, device=device)
idx[[0, 1, 2, 4, 5, 11]] = False


start=time.perf_counter()
X0 = torch.zeros_like(hard_element_force_train[0])
X0[idx, :] = torch.inverse(hard_element_stiffness_train[0][idx, :][:, idx])@hard_element_force_train[0][idx, :]
end=time.perf_counter()
print(end-start)


start=time.perf_counter()
CH_0=num_homo.homogenized(voxels[0],U,hard_element_stiffness_train[0],hard_element_force_train[0])
CH_0=shape_material_transform(CH_0,jacobi[0])
CH_1=num_homo.homogenized(voxels[0],U,ke,fe)
print(torch.dist(CH_0,CH_1))
end=time.perf_counter()
print(end-start)


