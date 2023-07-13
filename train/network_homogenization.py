
from numpy import dtype
from torch_scatter import segment_sum_csr
import torch

from torch_sparse import spmm

import time
import torch.distributed as dist

class network_homogenization:

    # rewrited by penghao

    def __init__(self, nelx, nely, nelz, lx, ly, lz,device) -> None:
       

        self.__K_indices = None
        self.__F_indices = None

        self.__anchor_indices = None
        self.__K_mask = None
        self.__F_mask = None

        self.__K_sortidx = None
        self.__K_ptr = None

        self.__F_sortidx = None
        self.__F_ptr = None

        self.__lx = lx
        self.__ly = ly
        self.__lz = lz
        self.__nelx = nelx
        self.__nely = nely
        self.__nelz = nelz
        self.device=device

        nel = nelx * nely * nelz
        nodeidx = torch.arange(0, nel,device=self.device).view(nelx, nely, nelz)

        index = torch.as_tensor([0],device=self.device)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 0, index)), 0)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 1, index)), 1)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 2, index)), 2)

        node_list = [nodeidx[0:nelx, 0:nely, 0:nelz].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 0:nely, 0:nelz].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 1:nely + 1, 0:nelz].reshape((nel, 1)),
                     nodeidx[0:nelx, 1:nely + 1, 0:nelz].reshape((nel, 1)),
                     nodeidx[0:nelx, 0:nely, 1:nelz + 1].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 0:nely, 1:nelz + 1].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 1:nely + 1,
                     1:nelz + 1].reshape((nel, 1)),
                     nodeidx[0:nelx, 1:nely + 1, 1:nelz + 1].reshape((nel, 1))]

        self.__cellidx = torch.zeros(
            8, 3, nel, device=self.device, dtype=torch.int64)
        self.__cellseq = torch.zeros(nel, 8, device=self.device, dtype=torch.int64)

        for i in range(8):
            self.__cellidx[i] = self.index2xyz(node_list[i])
            self.__cellseq[:, i] = node_list[i].view(-1)

        self.__nodeidx = nodeidx

        self.anchor()
        self.indices()
        self.__vK=torch.empty(nelx*nely*nelz,24,24,dtype=torch.float,device=self.device)
        self.__vF=torch.empty(nelx*nely*nelz,24,6,dtype=torch.float,device=self.device)


    @torch.no_grad()
    def index2xyz(self, index):
        x = index.div(self.__nely * self.__nelz, rounding_mode='floor')
        temp = index .remainder(self.__nely * self.__nelz)
        y = temp.div(self.__nely,rounding_mode='floor')
        z = temp.remainder( self.__nely)
        xyz = torch.cat((x, y, z), 1)
        return xyz.t()

    @torch.no_grad()
    def anchor(self, index=0):
        anchor = self.__nodeidx[self.__cellidx[index, 0, 0],
                                self.__cellidx[index, 1, 0], self.__cellidx[index, 2, 0]]

        mask = torch.eq(self.__cellseq, anchor)

        anchor_index = torch.arange(
            0, self.__nelx*self.__nely*self.__nelz, device=self.device)
        anchor_cell = anchor_index.masked_select(mask.sum(1).type(torch.bool))
        anchor_mask = mask[anchor_cell, :]

        if anchor_mask.dim() == 1:
            anchor_mask.unsqueeze_(0)

        anchor_mask = ~anchor_mask.unsqueeze(2).repeat(
            1, 1, 3).reshape(-1, 24).unsqueeze(2)
        anchor_mask = torch.as_tensor(
            anchor_mask, dtype=torch.float64, device=self.device)
        K_mask = anchor_mask.bmm(anchor_mask.transpose(1, 2))
        K_diag = torch.eye(24, 24, dtype=K_mask.dtype, device=self.device).unsqueeze(
            0).repeat(K_mask.shape[0], 1, 1)
        K_mask = torch.logical_or(K_mask, K_diag)
        F_mask = anchor_mask.repeat(1, 1, 6)

        self.__anchor_indices = torch.as_tensor(anchor_cell,dtype=torch.float,device=self.device)
        self.__K_mask = torch.as_tensor(K_mask,dtype=torch.float,device=self.device)
        self.__F_mask = torch.as_tensor(F_mask,dtype=torch.float,device=self.device)

        # return anchor_cell, K_mask, F_mask

    @torch.no_grad()
    def indices(self):
        n = self.__nelx * self.__nely * self.__nelz
        dof_indices = torch.empty(
            n, 8, 3, dtype=self.__cellseq.dtype, device=self.device)
        for i in range(3):
            dof_indices[:, :, i] = 3*self.__cellseq+i
        dof_indices = torch.as_tensor(
            dof_indices, dtype=torch.float, device= self.device).view(-1, 24).unsqueeze(2)

        # torch-sparse_solver version
        Kij = torch.zeros(2, 24 * 24 * n, device= self.device)
        temp = torch.ones(
            n, 24, 1, dtype=torch.float, device= self.device)
        Kij[0, :] = dof_indices.bmm(temp.transpose(1, 2)).contiguous().view(-1)
        Kij[1, :] = temp.bmm(dof_indices.transpose(1, 2)).contiguous().view(-1)

        Fij = torch.zeros(2, 24 * 6 * n, device= self.device)
        temp_F = torch.ones(n, 6, 1, dtype=torch.float, device= self.device)
        Fij[0, :] = dof_indices.bmm(
            temp_F.transpose(1, 2)).contiguous().view(-1)
        Fij[1, :] = torch.arange(0, 6, device= self.device).unsqueeze(0).unsqueeze(
            0).repeat(n, 24, 1).contiguous().view(-1)

        self.__K_indices = torch.as_tensor(Kij, dtype=torch.int64)
        self.__F_indices = torch.as_tensor(Fij, dtype=torch.int64)
       


    # def set_coalesce(self):
        nd = 3*self.__nelx * self.__nely * self.__nelz
        # coalesce+symmetry
        sorted_K,self.__K_sortidx=torch.cat([self.__K_indices[0]*nd+self.__K_indices[1], self.__K_indices[1]*nd+self.__K_indices[0]]).sort()
        sorted_K=torch.cat([-torch.ones(1,dtype=sorted_K.dtype,device=self.device),sorted_K])
        mask = sorted_K[1:] > sorted_K[:-1]
        self.__K_indices = self.__K_indices.repeat(1, 2)[:, self.__K_sortidx][:, mask]

        # print(self.__K_indices.shape)
        self.__K_ptr = mask.nonzero().flatten()
        self.__K_ptr = torch.cat([self.__K_ptr, self.__K_ptr.new_full((1, ),  mask.numel())])


        sorted_F,self.__F_sortidx=(self.__F_indices[0]*6+self.__F_indices[1]).sort()
        sorted_F=torch.cat([-torch.ones(1,dtype=sorted_F.dtype,device=self.device),sorted_F])
        mask = sorted_F[1:] >sorted_F[:-1]
        self.__F_indices = self.__F_indices[:, self.__F_sortidx][:, mask]
        self.__F_ptr = mask.nonzero().flatten()
        self.__F_ptr = torch.cat(
            [self.__F_ptr, self.__F_ptr.new_full((1, ),  mask.numel())])


    @torch.no_grad()
    def get_K_indices(self):
        return self.__K_indices

    @torch.no_grad()
    def symcoalesce(self, value):
        return segment_sum_csr(value.repeat(2)[self.__K_sortidx]*0.5, self.__K_ptr)
    
    @torch.no_grad()   
    def coalesce(self, value):
        value_ = segment_sum_csr(value[self.__F_sortidx], self.__F_ptr)
        return value_

    @torch.no_grad()
    def full_assembly(self, voxel, Ke_hard, Fe_hard, Ke_soft, Fe_soft):
        
        # voxel_=torch.as_tensor(voxel,dtype=torch.float,device=self.device).contiguous()
        
        # fast_repeat(voxel_,Ke_hard,Ke_soft,self.__vK)
        # fast_repeat(voxel_,Fe_hard,Fe_soft,self.__vF)
       
        # fast_anchor(self.__vK,self.__anchor_indices,self.__K_mask)
        # fast_anchor(self.__vF,self.__anchor_indices,self.__F_mask)

        ## By pytorch
        n=voxel.numel()
        n_hard = int(voxel.sum().item())
        n_soft = n-n_hard
        
        hard_mask = torch.as_tensor(
            voxel, dtype=torch.bool, device=voxel.device).contiguous().view(-1)

        self.__vK[hard_mask, :, :] = Ke_hard.repeat(n_hard, 1, 1)
        self.__vF[hard_mask, :, :] = Fe_hard.repeat(n_hard, 1, 1)
        self.__vK[~hard_mask, :, :] = Ke_soft.repeat(n_soft, 1, 1)
        self.__vF[~hard_mask, :, :] = Fe_soft.repeat(n_soft, 1, 1)


        # new anchor by set F(anchor)->0
        anchor_indices=torch.as_tensor(self.__anchor_indices,dtype=torch.int64)
        for idx in range(self.__K_mask.shape[0]):
            self.__vK[anchor_indices[idx], :, :] =  self.__vK[anchor_indices[idx], :, :] *  self.__K_mask[idx, :, :]
            self.__vF[anchor_indices[idx], :, :] =  self.__vF[anchor_indices[idx], :, :] *  self.__F_mask[idx, :, :]



       
        vK_=self.symcoalesce( self.__vK.contiguous().view(-1))
        vF_=self.coalesce( self.__vF.contiguous().view(-1)).view(-1,6)

        return vK_, vF_


    @torch.no_grad()
    def homogenized(self, voxel, U, ke_hard, X0):
        solid_seq = self.__cellseq[voxel.type(
            torch.bool).contiguous().view(-1), :]
        n = solid_seq.shape[0]

        volume = self.__lx * self.__ly * self.__lz

        # reshape U(18,N,N,N)->u(6,3,N^3),then transpose u(6,3,N^3)->u(6,N^3,3) 
        u_ = U.contiguous().view(6, 3, -1).transpose(1, 2)
        # reshape u(6,N^3,3)->u(6,N^3*3),transpose u(6,N^3*3)->u(N^3*3,6)
        u_ = u_.contiguous().view(6, -1).t()

        # index_u=self.__cellidx.movedim(2,0).contiguous().view(-1).to(U.device)

        index_u = torch.empty(n, 8, 3, dtype=torch.int64, device=self.device)
        for i in range(3):
            index_u[:, :, i] = 3 * solid_seq + i
        index_u = index_u.contiguous().view(-1)

        u = u_[index_u, :].contiguous().view(n, 24, 6)
        del index_u

        CH = torch.zeros(6, 6, dtype=ke_hard.dtype, device=self.device)
        L = X0-u
        CH = torch.einsum('bij,ik,bkl-> jl',L,ke_hard,L)
        return 1 / volume * CH


    @torch.no_grad()
    def MPE_full(self, voxel, Ke_hard, Fe_hard, Ke_soft, Fe_soft, U):
        """compute minimal potential energy according to 0.5*U^T@K@U-U^T@F

        Args:
           voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
           Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
           Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 
           U (torch.cuda.FloatTensor): it should be 18*N*N*N

        Returns:
            energy(float) :  minimal potential energy
        """
        nd = 3*self.__nelx * self.__nely * self.__nelz
        
        tic=time.perf_counter()
       
        vK, F = self.full_assembly(voxel, Ke_hard, Fe_hard, Ke_soft, Fe_soft)

        toc=time.perf_counter()
        # if dist.get_rank()==3:
        #      print(self.device,'   {:.4f}--------\n'.format((toc-tic)*1000))

       
        u = U.contiguous().view(6, 3, -1).transpose(1, 2)
        u = u.contiguous().view(6, -1).t()

        Ku=spmm(self.__K_indices,vK,nd,nd,u)
        
        grad = (Ku - F).t().contiguous().view(6, -1, 3).transpose(1,
                                                                  2).contiguous().view(18, self.__nelx, self.__nely, self.__nelz)
      
        energy = ((u.t() @ (0.5*Ku - F)) *
                  torch.eye(6, 6, dtype=u.dtype, device=self.device)).sum()  # .item()
        
        return energy, grad
    
    @torch.no_grad()
    def compute_ke_fe(self, C):
        """Compute stiffness and load matrix

        Args:
            C (_type_):Base material

        Returns:
            Ke: stiffness matrix per element
            Fe: Load matrix per element
        """
        dx = self.__lx / self.__nelx / 2
        dy = self.__ly / self.__nely / 2
        dz = self.__lz / self.__nelz / 2

        pp = torch.as_tensor(
            [-pow(3 / 5, 0.5), 0, pow(3 / 5, 0.5)], dtype=C.dtype, device=self.device)
        ww = torch.as_tensor([5 / 9, 8 / 9, 5 / 9],
                             dtype=C.dtype, device=self.device)
        Ke = torch.zeros(24, 24, dtype=C.dtype, device=self.device)
        Fe = torch.zeros(24, 6, dtype=C.dtype, device=self.device)

        dxdydz = torch.as_tensor(
            [[-dx, dx, dx, -dx, -dx, dx, dx, -dx], [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
             [-dz, -dz, -dz, -dz, dz, dz, dz, dz]], dtype=C.dtype, device=self.device).t()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = pp[i]
                    y = pp[j]
                    z = pp[k] 
                    qxqyqz = torch.as_tensor(
                        [[-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                          ((y + 1) * (z - 1)) / 8, ((y - 1) *
                                                    (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                          ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8],
                         [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                          ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) /
                          8, -((x + 1) * (z + 1)) / 8,
                          ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8],
                         [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                          ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) /
                          8, -((x + 1) * (y - 1)) / 8,
                          ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]], dtype=C.dtype, device=self.device)

                    J = qxqyqz @ dxdydz
                    invJ = torch.inverse(J)
                    qxyz = invJ @ qxqyqz
                    B = torch.zeros(6, 24, dtype=C.dtype, device=self.device)

                    for i_B in range(8):
                        B[:, i_B * 3:(i_B + 1) * 3] = torch.as_tensor(
                            [[qxyz[0, i_B], 0, 0],
                             [0, qxyz[1, i_B], 0],
                             [0, 0, qxyz[2, i_B]],
                             [qxyz[1, i_B],
                              qxyz[0, i_B], 0],
                             [0, qxyz[2, i_B],
                              qxyz[1, i_B]],
                             [qxyz[2, i_B], 0, qxyz[0, i_B]]], dtype=C.dtype,
                            device=self.device)

                    weight = J.det() * ww[i] * ww[j] * ww[k]

                    Ke = Ke + weight * B.transpose(0, 1) @ C @ B
                    Fe = Fe + weight * B.transpose(0, 1) @ C

        return Ke, Fe

    
