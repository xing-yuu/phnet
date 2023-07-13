import torch
import torch.sparse as ts
from torch_sparse import coalesce
from linear_operator.utils.linear_cg import linear_cg
import numpy as np


class numerical_homogenization:

    # rewrited by penghao

    def __init__(self, nelx, nely, nelz, lx, ly, lz, device) -> None:

        self.device=device
        self.__lx = lx
        self.__ly = ly
        self.__lz = lz
        self.__nelx = nelx
        self.__nely = nely
        self.__nelz = nelz
        nel = nelx * nely * nelz
        nodeidx = torch.arange(nel, device=self.device).view(nelx, nely, nelz)

        index = torch.as_tensor([0], device=self.device)
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

        cell_dof_indices = torch.zeros(nel, 8*3, device=self.device, dtype=torch.int64)
        for i in range(8):
            self.__cellidx[i] = self.index2xyz(node_list[i])
            self.__cellseq[:, i] = node_list[i].view(-1)
            for j in range(3):
                cell_dof_indices[:,i*3+j]=(node_list[i]*3+j).reshape(-1)
            
        self.__nodeidx = nodeidx

    def printindex(self):
        print(self.__cellidx.size())
        return self.__cellidx

    def index2xyz(self, index):
        x = index.div(self.__nely * self.__nelz, rounding_mode='floor')
        temp = index .remainder(self.__nely * self.__nelz)
        y = temp.div(self.__nely,rounding_mode='floor')
        z = temp.remainder( self.__nely)
        xyz = torch.cat((x, y, z), 1)
        return xyz.t()

    # rewrited by penghao
    def solid_cell(self, voxel):
        """return solid cell index from voxel

        Args:
            voxel (torch.cuda.FloatTensor): it should be N*N*N 

        Returns:
            solid_cell (torch.cuda.IntTensor): it should be 8*3*nel 
            ancher_cell(torch.cuda.FloatTensor): it should be a 
            K_mask(torch.cuda.FloatTensor): it should be a*24*24 
            F_mask(torch.cuda.FloatTensor): it should be a*24*6 
        """
        # print(voxel.device)
        voxelidx = torch.arange(0, voxel.numel(), device=self.device)
        voxelidx = torch.masked_select(
            voxelidx, voxel.type(torch.bool).contiguous().view(-1))
        solid_cell = self.__cellidx[:, :, voxelidx]
        solid_seq = self.__cellseq[voxelidx, :]
        anchor = self.__nodeidx[solid_cell[0, 0, 0],
                                solid_cell[0, 1, 0], solid_cell[0, 2, 0]]
        # print(solid_cell[0, 0, 0], solid_cell[0, 1, 0], solid_cell[0, 2, 0])
        # print(anchor)
        mask = torch.eq(solid_seq, anchor)

        anchor_index = torch.arange(
            0, voxelidx.numel(), device=self.device)
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
        return solid_cell, anchor_cell, K_mask, F_mask

   
    def hexahedron(self, C):
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

    def homo(self, voxel, disp, elastic_tensor):

        volume = self.__lx * self.__ly * self.__lz

        solid_cell, _, _, _ = self.solid_cell(voxel)
        num = solid_cell.shape[0] * solid_cell.shape[2]
        x = solid_cell[:, 0, :].t().reshape(num)
        y = solid_cell[:, 1, :].t().reshape(num)
        z = solid_cell[:, 2, :].t().reshape(num)

        solid_num = solid_cell.shape[2]
        U = torch.zeros(solid_num, 24, 6).type(torch.DoubleTensor)
        for j in range(6):
            u = disp[j * 3:(j + 1) * 3, x, y, z].t()
            U[:, :, j] = u.reshape(solid_num, 24)

        I = torch.eye(6, 6).type(torch.DoubleTensor).to(self.device)
       

        C = elastic_tensor.type(torch.DoubleTensor).to(self.device)
       

        CH = torch.zeros(6, 6).type(torch.DoubleTensor).to(self.device)

        dx = self.__lx / self.__nelx / 2
        dy = self.__ly / self.__nely / 2
        dz = self.__lz / self.__nelz / 2

        pp = torch.Tensor([-pow(3 / 5, 0.5), 0, pow(3 / 5, 0.5)]).to(self.device)
        ww = torch.Tensor([5 / 9, 8 / 9, 5 / 9]).to(self.device)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = pp[i]
                    y = pp[j]
                    z = pp[k]
                    qx = torch.Tensor([-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                                       ((y + 1) * (z - 1)) / 8, ((y - 1) *
                                                                 (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                                       ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8]).to(self.device)
                    qy = torch.Tensor([-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                                       ((x - 1) * (z - 1)) / 8, ((x - 1) *
                                                                 (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,
                                       ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8]).to(self.device)
                    qz = torch.Tensor([-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                                       ((x - 1) * (y + 1)) / 8, ((x - 1) *
                                                                 (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,
                                       ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]).to(self.device)

                    qxqyqz = torch.cat((qx, qy, qz), 0).view(
                        [-1, 8]).to(self.device)
                    J = qxqyqz @ torch.Tensor(
                        [[-dx, dx, dx, -dx, -dx, dx, dx, -dx], [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
                         [-dz, -dz, -dz, -dz, dz, dz, dz, dz]]).t().to(self.device)
                    invJ = torch.inverse(J).to(self.device)
                    qxyz = invJ @ qxqyqz

                    B = torch.zeros(6, 0).to(self.device)

                    for i_B in range(8):
                        Be = torch.tensor([[qxyz[0, i_B], 0, 0],
                                           [0, qxyz[1, i_B], 0],
                                           [0, 0, qxyz[2, i_B]],
                                           [qxyz[1, i_B], qxyz[0, i_B], 0],
                                           [0, qxyz[2, i_B], qxyz[1, i_B]],
                                           [qxyz[2, i_B], 0, qxyz[0, i_B]]]).to(self.device)
                        B = torch.cat((B, Be), 1)

                    B = B.type(torch.DoubleTensor).to(self.device)

                    weight = J.det() * ww[i] * ww[j] * ww[k]

                    # L=I-B@U
                    # L = torch.baddbmm(I, B, U, alpha=-1)
                    L=I-B@U.to(self.device)
                    # CH=CH+weight*(I-B @ U)^T @ C @ (I-B @ U)
                    CH = torch.addbmm(CH, L.transpose(
                        1, 2), C.to(self.device)@L.to(self.device), alpha=weight)

        return 1 / volume * CH
    
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


    def assembly(self, voxel, Ke, Fe):
        """assemble stiffness matrix K and force matrix for solid cell

        Args:
           voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
           Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
           Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 

        Returns:
            K(torch.cuda.Soarse_COO_Tensor)
            F(torch.cuda.FloatTensor)
        """
        voxelidx = torch.arange(0, voxel.numel(),device=self.device)
        voxelidx = torch.masked_select(
            voxelidx, voxel.type(torch.bool).contiguous().view(-1))

        solid_seq = self.__cellseq[voxelidx, :]

        _, anchor_cell, K_mask, F_mask = self.solid_cell(voxel)

        indices, inv_indices = torch.unique(
            solid_seq, sorted=True, return_inverse=True)

        nv = indices.shape[0]
        nc = inv_indices.shape[0]

        dof = torch.zeros(nv, 3, device=self.device)
        dof_indices = torch.zeros(
            nc, 24, device=self.device, dtype=torch.int64)

        for i in range(3):
            dof[:, i] = 3 * indices + i
            dof_indices[:, torch.arange(0, 8) * 3 + i] = inv_indices * 3 + i

        dof = torch.as_tensor(dof.contiguous().view(-1),
                              dtype=torch.int64, device=self.device)
        dof_indices = torch.as_tensor(
            dof_indices.unsqueeze(2), dtype=Ke.dtype, device=self.device)

        vK = Ke.repeat(nc, 1, 1)
        vF = Fe.repeat(nc, 1, 1)
        # new anchor by set F(anchor)->0
        for idx in range(K_mask.shape[0]):
            vK[anchor_cell[idx], :, :] = vK[anchor_cell[idx], :, :] * \
                                         K_mask[idx, :, :]
            vF[anchor_cell[idx], :, :] = vF[anchor_cell[idx], :, :] * \
                                         F_mask[idx, :, :]

        # torch-sparse_solver version
        Kij = torch.zeros(2, 24 * 24 * inv_indices.shape[0], device=self.device)
        temp = torch.ones(
            inv_indices.shape[0], 24, dtype=Ke.dtype, device=self.device).unsqueeze(2)
        Kij[0, :] = dof_indices.bmm(temp.transpose(1, 2)).contiguous().view(-1)
        Kij[1, :] = temp.bmm(dof_indices.transpose(1, 2)).contiguous().view(-1)

        K = torch.sparse_coo_tensor(Kij, vK.contiguous(
        ).view(-1), (3 * nv, 3 * nv), device=self.device).coalesce()

        
        Kij,_=coalesce(torch.as_tensor(Kij,dtype=torch.int64),vK.contiguous(
        ).view(-1),3*nv,3*nv)

        # print(Kij.shape)
        # print(K.indices().shape)

        Fij = torch.zeros(2, 24 * 6 * inv_indices.shape[0], device=self.device)
        temp_F = torch.ones(
            inv_indices.shape[0], 6, dtype=Ke.dtype, device=self.device).unsqueeze(2)
        Fij[0, :] = dof_indices.bmm(
            temp_F.transpose(1, 2)).contiguous().view(-1)
        Fij[1, :] = torch.arange(0, 6, device=self.device).unsqueeze(0).unsqueeze(
            0).repeat(inv_indices.shape[0], 24, 1).contiguous().view(-1)
        F = torch.sparse_coo_tensor(Fij, vF.contiguous(
        ).view(-1), (3 * nv, 6), device=self.device).coalesce().to_dense()

        return K, F, dof

 
    def solve_by_torch(self, voxel, Ke, Fe, tol=1e-3, maxit=5000):
        """solve Ku=f by pytorch using linear conjugate gradient method

        Args:
            voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
            Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
            Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 
            tol (float, optional): Linear cg tolerance. Defaults to 1e-3.
            maxit (int, optional): Maximum iteration number. Defaults to 1000.

        Returns:
            U (torch.cuda.FloatTensor): it should be 18*N*N*N
        """

        K, F, dof = self.assembly(voxel, Ke, Fe)

        def Kmm(rhs): return ts.mm(K, rhs)

        X = linear_cg(Kmm, F, tolerance=tol, max_iter=maxit)

        u = torch.zeros(3 * self.__cellseq.shape[0], 6,dtype=X.dtype,device=self.device)
        u[dof, :] = X

        # transpose u(6,N^3*3)<-u(N^3*3,6),reshape u(6,N^3,3)-<u(6,N^3*3)
        u = u.t().contiguous().view(6, -1, 3)
        # transpose u(6,3,N^3)<-u(6,N^3,3),reshape U(18,N,N,N)<-u(6,3,N^3),
        u = u.transpose(1, 2).contiguous().view(
            18, self.__nelx, self.__nely, self.__nelz)
        return u


        

   

