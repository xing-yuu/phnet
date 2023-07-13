from tg import *
import numpy as np
import torch
from train.homogenization_helper import *
from train.numerical_homogenization import *
import os
import random
import argparse


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

res=48
voxel_nbr=40

tg=TG(res,1)

voxel=np.empty((voxel_nbr,res,res,res))

thr=np.linspace(6,13.8,voxel_nbr)

for i,t in enumerate(thr):
   voxel[i]=tg.voxel(t)


if __name__=='__main__': 

    setup_seed(100)
    #arguments
    parser=argparse.ArgumentParser(description="DH-Net dataset generator")

    parser.add_argument('config', type=str, help='Path to config file.')

    args=parser.parse_args()
    # with open(args.config, 'configs/generate_default.yaml')
    cfg=config.load_config(args.config,'configs/default.yaml')

    out_dir=cfg['dataset']['out_dir']
    

E=1
v=0.3
device=torch.device('cuda')
C=isotropic_elastic_tensor(E,v).to(device)

if not os.path.exists(out_dir):
      os.makedirs(out_dir)

H=numerical_homogenization(res,res,res,1,1,1,device)
Ke,Fe=H.hexahedron(C)

