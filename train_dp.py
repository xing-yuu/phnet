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
from train.dataset import *
from train.eval import *
from train.minimal_potential_energy_loss import *
from train.network_homogenization import *
from train.numerical_homogenization import *
from train.unet_full_tensor import *
from train.homogenization_helper import *
from tracker.gpu_mem_track import MemTracker


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

@torch.no_grad()   
def homogenized_material(test_loader,net,net_homo,ke,X0):
    net.eval()
    input, voxels, hard_element_stiffness, hard_element_force,_,_,_,_,_= next(iter(test_loader))
    output = net(input)
    C_homo=net_homo.homogenized(voxels[0],output[0],ke,X0)
    return C_homo

@torch.no_grad()   
def validate(test_loader,net,net_homo,numer_homo,ke,X0):
    net.eval()
    input, voxels, hard_element_stiffness, hard_element_force,_,_,_,_,_= next(iter(test_loader))
    output = net(input)
    batch_error_mean, batch_error = eval(net_homo,numer_homo, voxels, output, hard_element_stiffness, hard_element_force,ke,X0)
    print(batch_error_mean)
    print('validation done!')
    
    # test prediction
    
    
    return batch_error_mean,batch_error


def ph_net(my_config):


  
    setup_seed(0)

    cfg=config.load_config(my_config,'configs/default.yaml')
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    

    ##dataset
    dataset_dir=cfg['dataset']['out_dir']
    train_ratio=cfg['dataset']['train_ratio']
    r=cfg['dataset']['voxel']['resolution']
    sample_per_voxel=cfg['dataset']['shape']['sample_per_voxel']
    out_dir=cfg['train']['out_dir']

    train_dataset,test_dataset=load_dataset(dataset_dir,r,sample_per_voxel,train_ratio,device)

    batch_size=cfg['train']['batch_size']
    shuffle=cfg['dataset']['shuffle']
    # eval_interval=int(cfg['train']['eval_interval'])
    # saving_interval=int(cfg['train']['saving_interval'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=shuffle,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=shuffle,drop_last=True)

    ## initialize homogenization solver
    ## numerical solver
    cache_file='train/cache/fem_cache.npz'
    homo_helper=homogenization_helper(cache_file,device)

    E=float(cfg['dataset']['material']['youngs_modulus_hard'])
    v=float(cfg['dataset']['material']['poisson_ratio'])
    elastic_tensor=isotropic_elastic_tensor(E,v).to(device)
    ke,X0=homo_helper.macro_deformation(r,r,r,elastic_tensor)

    numer_homo=numerical_homogenization(r,r,r,1,1,1,device)
    ## network solver
    net_homo=network_homogenization(r,r,r,1,1,1,device)

    ##network
    lr=float(cfg['train']['learning_rate'])
    epochs=int(cfg['train']['epoch'])
    pre_train_model=cfg['train']['pre_train']
    
    net = U_Net()
    if not pre_train_model==None:
        if (not len(pre_train_model)==0) and os.path.isfile(os.path.join(out_dir,'model.pt')):
            net.load_state_dict(torch.load(os.path.join(out_dir,'model.pt')))
            print('test1')
            net.train()
            print('test2')
            # checkpoint = torch.load(os.path.join(out_dir,'model.pt'))
            # print(checkpoint)
            
            # net.load_state_dict(checkpoint['state_dict'])
            # net.train()
     
    net = torch.nn.DataParallel(net,device_ids = [0,1, 2, 3])
    net=net.to(device)      
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_func = MPELoss_with_backward.apply

   
    ##train
    tbdir=None
    writer=None
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tbdir= os.path.join(out_dir,'logs')
    writer= SummaryWriter(tbdir)
    
    for e in tqdm(range(epochs)):
        iter=tqdm(train_loader)
        tqdm_str=''
        print('\nStart validation')
        batch_error_mean, batch_error = validate(test_loader,net,net_homo,numer_homo,ke,X0)   
        tqdm_str+=', mean error={:+.6f}'.format(batch_error_mean)

        writer.add_scalar('Error_Avg', batch_error_mean, e*len(train_loader))
        for i in range(len(batch_error)):
            scalar_name = 'Error' + str(i)
            writer.add_scalar(scalar_name, batch_error[i], e*len(train_loader))

        
        for step, (input, voxels, hard_element_stiffness, hard_element_force, soft_element_stiffness, soft_element_force,_,_,_) in enumerate(iter):
            
            output = net(input)
            tic=time.perf_counter()
            loss = loss_func(output, net_homo, voxels, hard_element_stiffness, hard_element_force, soft_element_stiffness, soft_element_force)
            toc=time.perf_counter()
           
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            
            ## Write log and save check point
            tqdm_str= '[GPU {:<2d}] Epoch ={:<2d} | loss={:+.6f}'.format(0,e,loss.item())
            writer.add_scalar('Loss', loss.item(), step+e*len(train_loader))
            
            time_cost=(toc-tic)*1000
            tqdm_str+=', time cost ={:+.2f}'.format(time_cost)
            iter.set_description(tqdm_str)
            
            # if step>0 and step % eval_interval == 0 :
            #     batch_error_mean, batch_error = eval(net_homo,numer_homo, voxels, output, hard_element_stiffness, hard_element_force,ke,X0)
            #     tqdm_str+=', mean error={:+.6f}'.format(batch_error_mean)

            #     writer.add_scalar('Error_Avg', batch_error_mean, step+e*len(train_loader))
            #     for i in range(len(batch_error)):
            #         scalar_name = 'Error' + str(i)
            #         writer.add_scalar(scalar_name, batch_error[i], step+e*len(train_loader))

        # if e % saving_interval==0:    
        #     torch.save({ 'model_state_dict': net.module.state_dict()}
        #         ,os.path.join(out_dir,'model.pt'))
        #     print('Saving Checkpoint done!')
        
        
        torch.save(net.module.state_dict(),os.path.join(out_dir,'model.pt'))
        print('Saving Checkpoint done!')
        net.train()


if __name__=='__main__':
    
    parser=argparse.ArgumentParser(description="DH-Net dataset generator")

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--world_size', type=int, default=4,
                    help='Number of visible GPU')

    args=parser.parse_args()
    ph_net(args.config)

   
    

    


