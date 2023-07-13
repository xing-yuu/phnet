import torch 
import math as m

def shear_transform(angle):

    angle_=torch.deg2rad(angle)

    H=1./torch.tan(angle_)

    S=angle.repeat(3,1)*0
    S[0,0]=1
    S[1,1]=1
    S[2,2]=1
    S[0,1]=H[0]
    S[0,2]=H[2]
    S[1,2]=H[1]

    # S=torch.tensor([[1,H[0],H[2]],[0,1,H[1]],[0,0,1]],device=angle.device)
    return S

def scale_transform(t):
    return torch.diag(t/t.prod().pow(1./3.))

                                                                                   