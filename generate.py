import logging
import os
import config
import random
import numpy as np
import torch
import argparse
from datagen import shear_transform,scale_transform,stiffness_force,shape_material_transform,isotropic_elastic_tensor
from scipy.stats import qmc

def LHC_sampler(dim:int,l_bnds:np.ndarray,u_bnds:np.ndarray):
    lhc_sampler = qmc.LatinHypercube(d=dim) 
    def sample(n:int):
        samples=lhc_sampler.random(n)
        return qmc.scale(samples,l_bnds,u_bnds)   
    return sample

logger = logging.getLogger(__name__)

def generator(scale,angle,C_hard,C_soft):
    
    J=shear_transform(angle)@scale_transform(scale)
    inv_J=J.inverse()

    C_hard_=shape_material_transform(C_hard,inv_J)
    C_soft_=shape_material_transform(C_soft,inv_J)

    hex=torch.tensor([[0,0,0],[1.,0,0],[1.,1.,0],[0,1.,0],[0,0,1.],[1.,0,1.],[1.,1.,1.],[0,1.,1.]],dtype=C_hard.dtype,device=C_hard.device)/36

    Ke_hard,Fe_hard=stiffness_force(hex,C_hard_)
    Ke_soft,Fe_soft=stiffness_force(hex,C_soft_)

    return Ke_hard,Fe_hard,Ke_soft,Fe_soft,C_hard_,C_soft_,J

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__=='__main__': 

    setup_seed(100)
    #arguments
    parser=argparse.ArgumentParser(description="DH-Net dataset generator")

    parser.add_argument('config', type=str, help='Path to config file.')

    args=parser.parse_args()
    # with open(args.config, 'configs/generate_default.yaml')
    cfg=config.load_config(args.config,'configs/default.yaml')

    out_dir=cfg['dataset']['out_dir']
    
    ##material 
    E_hard=float(cfg['dataset']['material']['youngs_modulus_hard'])
    E_soft=float(cfg['dataset']['material']['youngs_modulus_soft'])
    v=float(cfg['dataset']['material']['poisson_ratio'])

    ##voxel
    voxel_nbr=cfg['dataset']['voxel']['size']
    res=cfg['dataset']['voxel']['resolution']
    voxel_dir=os.path.join(cfg['dataset']['voxel']['dir'],'voxel.npz')

    ##shape
    sample_per_voxel=cfg['dataset']['shape']['sample_per_voxel']
    scale_min=cfg['dataset']['shape']['scale_min']
    scale_max=cfg['dataset']['shape']['scale_max']
    angle_min=cfg['dataset']['shape']['angle_min']
    angle_max=cfg['dataset']['shape']['angle_max']

    sample_nbr=sample_per_voxel*voxel_nbr
    print(np.hstack([(scale_min,)*3,(angle_min,)*3]))
    sample=LHC_sampler(6,np.hstack([(scale_min,)*3,(angle_min,)*3]),np.hstack([(scale_max,)*3,(angle_max,)*3]))
    
    shape_param=[sample(sample_per_voxel) for i in range(voxel_nbr)]
    shape_param=np.vstack(shape_param)
    

    if not os.path.exists(out_dir):
        logger.error(f'Make directory: {out_dir}')
        os.makedirs(out_dir)


    C_hard=isotropic_elastic_tensor(E_hard,v).cuda()
    C_soft=isotropic_elastic_tensor(E_soft,v).cuda()

    C_hard=(C_hard.unsqueeze(0)).repeat(sample_nbr,1,1)
    C_soft=(C_soft.unsqueeze(0)).repeat(sample_nbr,1,1)

    # scale=(torch.rand(sample_nbr,3)*(scale_max-scale_min)+scale_min).cuda()
    # angle=(torch.rand(sample_nbr,3)*(angle_max-angle_min)+angle_min).cuda()
    scale=torch.from_numpy(shape_param[:,:3]).cuda()
    angle=torch.from_numpy(shape_param[:,3:]).cuda()
    
    Ke_hard=torch.empty(sample_nbr,24,24).cuda()
    Fe_hard=torch.empty(sample_nbr,24,6).cuda()

    Ke_soft=torch.empty_like(Ke_hard)
    Fe_soft=torch.empty_like(Fe_hard)
    J=torch.empty(sample_nbr,3,3)
    MA=torch.empty(sample_nbr,6,6)

    # vmap(generator)(scale,angle,C_hard,C_soft,MA)
    voxel=np.empty((voxel_nbr,res,res,res))
    if not os.path.isfile(voxel_dir):
        logger.error('Could not open {}!'.format(voxel_dir)) 
    else:
        voxel=np.load(voxel_dir)['voxel']
        
    for i in range(sample_nbr):
        print(f"Dataset generating {i}-th samples with scale {scale[i].cpu().numpy()} and angle {angle[i].cpu().numpy()}")
        
        Ke_hard[i],Fe_hard[i],Ke_soft[i],Fe_soft[i],C_hard[i],C_soft[i],J[i]=generator(scale[i],angle[i],C_hard[i],C_soft[i])

    
    np.savez(os.path.join(out_dir,'data.npz'),
            voxel=voxel,
            Ke_hard=Ke_hard.cpu().numpy(),
            Ke_soft=Ke_soft.cpu().numpy(),
            Fe_hard=Fe_hard.cpu().numpy(),
            Fe_soft=Fe_soft.cpu().numpy(),
            C_hard=C_hard.cpu().numpy(),
            C_soft=C_soft.cpu().numpy(),
            J=J.cpu().numpy())

    print('Done!')
    
  









