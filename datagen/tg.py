from cmath import pi
import pyvista as pv
import numpy as np

class TG:
    def __init__(self,res,period) -> None:

        r=res*period+1
        self.grid = pv.UniformGrid()

        self.grid.dimensions = (r,r,r)

        self.r=r-1
        m=0.75*np.pi
        s=2*np.pi/res

        self.grid.origin=(m,m,m)
        self.grid.spacing=(s,s,s)

        self.TG =lambda x,y,z: 10*(np.cos(x)*np.sin(y)+np.cos(y)*np.sin(z)+np.cos(z)*np.sin(x))-\
        0.5*(np.cos(2*x)*np.cos(2*y)+np.cos(2*y)*np.cos(2*z)+np.cos(2*z)*np.cos(2*x))


    def voxel(self,threshold):
        p=self.grid.cell_centers().points
        values=self.TG(p[:,0],p[:,1],p[:,2])

        print(p.min()/pi,p.max()/pi)
        voxel=np.zeros_like(values)
        voxel[values>=threshold]=1
        return voxel.reshape(self.r,self.r,self.r)