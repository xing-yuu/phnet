from tg import *
import numpy as np

res=36
voxel_nbr=40

tg=TG(res,1)

voxel=np.empty((voxel_nbr,res,res,res))

thr=np.linspace(6,13.8,voxel_nbr)

for i,t in enumerate(thr):
   voxel[i]=tg.voxel(t)


np.savez('voxel.npz',voxel=voxel)

