import os
import logging
import time
import torch
import random
import datetime
import argparse
from contextlib import contextmanager

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import config
from train.dataset_for_mango_net import *
from train.eval import *
from train.minimal_potential_energy_loss import *
from train.network_homogenization import *
from train.numerical_homogenization import *
from train.unet_isotropic_material_2 import *
from train.homogenization_helper import *
from tracker.gpu_mem_track import MemTracker

os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

torch.set_printoptions(
    precision=4,    # 精度，保留小数点后几位，默认4
    threshold=200,
    edgeitems=100,
    linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    sci_mode=True  # 用科学技术法显示数据，默认True
)


def SF(homo_helper,C_hard,C_soft,res,batch_size):
    
    hex=torch.tensor([[0,0,0],[1.,0,0],[1.,1.,0],[0,1.,0],[0,0,1.],[1.,0,1.],[1.,1.,1.],[0,1.,1.]],dtype=C_hard.dtype,device=C_hard.device)/res
    Ke_hard,Fe_hard=homo_helper.stiffness_force(hex,C_hard)
    Ke_soft,Fe_soft=homo_helper.stiffness_force(hex,C_soft)
    return Ke_hard.expand(batch_size,-1,-1),Fe_hard.expand(batch_size,-1,-1),Ke_soft.expand(batch_size,-1,-1),Fe_soft.expand(batch_size,-1,-1)

def dh_net(my_config):
    print("==================================Training Informtion==========================")
    
    setup_seed(2022)

    cfg=config.load_config(my_config,'configs/default.yaml')
   
    device = torch.device('cuda:1')


    ##dataset
    dataset_dir=cfg['dataset']['out_dir']
    train_ratio=cfg['dataset']['train_ratio']
    r=cfg['dataset']['voxel']['resolution']
    out_dir=cfg['train']['out_dir']

    batch_size=cfg['train']['batch_size']
    shuffle=cfg['dataset']['shuffle']
    eval_interval=int(cfg['train']['eval_interval'])
    saving_interval=int(cfg['train']['saving_interval'])
    
    
    cache_file='train/cache/fem_cache.npz'
    homo_helper=homogenization_helper(cache_file,device)

    E=float(cfg['dataset']['material']['youngs_modulus_hard'])
    E_soft=float(cfg['dataset']['material']['youngs_modulus_soft'])
    v=float(cfg['dataset']['material']['poisson_ratio'])

    base_material=torch.tensor([[E,v],[E_soft,v]]).to(device)

    print(base_material)

    train_dataset,_=load_dataset(dataset_dir,base_material,train_ratio,device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=shuffle,drop_last=True)

    print('Loading dataset done!')

    ## initialize homogenization solver

    elastic_tensor=isotropic_elastic_tensor(E,v).to(device)
    elastic_tensor_soft=isotropic_elastic_tensor(E_soft,v).to(device)

    hard_element_stiffness, hard_element_force, soft_element_stiffness, soft_element_force=SF(homo_helper,elastic_tensor,elastic_tensor_soft,r,batch_size)
    ke,X0=homo_helper.macro_deformation(r,r,r,elastic_tensor)


    ## numerical solver
    numer_homo=numerical_homogenization(r,r,r,1,1,1,device)
    ## network solver
    net_homo=network_homogenization(r,r,r,1,1,1,device)

    print('Initialize numerical homogenization and network homogenization done!')

    ##network
    lr=float(cfg['train']['learning_rate'])
    epochs=int(cfg['train']['epoch'])
    milestone=(cfg['train']['milestone'])
    pre_train_model=cfg['train']['pre_train']
    
    net = U_Net()
   
    if not pre_train_model==None:
        if (not len(pre_train_model)==0) and os.path.isfile(os.path.join(out_dir,'model.pt')):
            
            checkpoint = torch.load(os.path.join(out_dir,'model.pt'))
            net.load_state_dict(checkpoint['model_state_dict'])
            net.train()
     
    net = torch.nn.DataParallel(net,device_ids = [1, 2, 3])
    net=net.to(device)      
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=milestone,eta_min=lr*1e-2)

    loss_func = MPELoss_with_backward.apply
   
    print('Initialize MANGO-NET done!')
   
    ##train
    tbdir=None
    writer=None
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tbdir= os.path.join(out_dir,'log', 'log')
    writer= SummaryWriter(tbdir)

    for e in tqdm(range(epochs)):
        iter=tqdm(train_loader)
        for step, (input, voxels) in enumerate(iter):
            
            output = net(input)
            tic=time.perf_counter()
            loss = loss_func(output, net_homo, voxels, hard_element_stiffness, hard_element_force, soft_element_stiffness, soft_element_force)
            toc=time.perf_counter()
           
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            tqdm_str=''
            
            ## Write log and save check point
            tqdm_str= '[GPU {:<2d}] Epoch ={:<2d} | loss={:+.6f} | | lr={:+.6f}'.format(0,e,loss.item(),scheduler.get_last_lr()[0])
            writer.add_scalar('Loss', loss.item(), step+e*len(train_loader))
            
            if step>0 and step % eval_interval == 0 :
                batch_error_mean, batch_error = eval(net_homo,numer_homo, voxels, output, hard_element_stiffness, hard_element_force,ke,X0)
                tqdm_str+=', mean error={:+.6f}'.format(batch_error_mean)

                writer.add_scalar('Error_Avg', batch_error_mean, step+e*len(train_loader))
                for i in range(len(batch_error)):
                    scalar_name = 'Error' + str(i)
                    writer.add_scalar(scalar_name, batch_error[i], step+e*len(train_loader))

            
            time_cost=(toc-tic)*1000
            tqdm_str+=', time cost ={:+.2f}'.format(time_cost)
            iter.set_description(tqdm_str)

        if e % saving_interval==0:    
            torch.save({ 'model_state_dict': net.module.state_dict()}
                ,os.path.join(out_dir,'model.pt'))
            print('Saving Checkpoint done!')

if __name__=='__main__':
    
    
    parser=argparse.ArgumentParser(description="DH-Net dataset generator")

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--world_size', type=int, default=4,
                    help='Number of visible GPU')

    args=parser.parse_args()
    dh_net(args.config)

   
    

    


