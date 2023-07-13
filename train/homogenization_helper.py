import numpy as np
import torch

class homogenization_helper:

    def __init__(self, cache_file, device) -> None:

        cache = np.load(cache_file)
        self.partial_N = torch.from_numpy(cache['partial_N']).to(device)
        self.weights = torch.from_numpy(cache['weight']).to(device)
        self.device = device

 

    def jacobian_matrix(self, hex, volume):
        # coumpute jacobian 27 * 3 * 3
        J = torch.matmul(self.partial_N, hex)
        # compute weight 27*1
        w = (J.det().unsqueeze(1)*self.weights).unsqueeze(2)
        return (J*w).sum(0)*2/volume

    def stiffness_force(self, hex, elastic_tensor):
        """compute the stiffness and force of a hex element 

        Args:
            hex ([type]): 8 * 3
            elastic_tensor ([type]): 6 * 6

        Returns:
            stiffness [type]: 24*24
            force [type] 24*6
        """

        # coumpute jacobian 27 * 3 * 3
        J = torch.matmul(self.partial_N.type_as(elastic_tensor), hex)

        # compute weight 27*1
        w = (J.det().unsqueeze(1)*self.weights).unsqueeze(2).type_as(elastic_tensor)

        # compute the inverse of partial N 27 * 3 * 8
        inv_J = J.inverse()
        inv_N = (inv_J@self.partial_N.type_as(elastic_tensor))

        # assigned local geometry matrix B 27*6*24
        B = inv_N.unsqueeze(1).repeat(1, 6, 1, 1)*0
        idx_B = torch.as_tensor([[0, 1, 2, 3, 3, 4, 4, 5, 5], [
                                0, 1, 2, 0, 1, 1, 2, 0, 2]], dtype=torch.int64, device=hex.device)
        idx_inv_N = torch.as_tensor(
            [0, 1, 2, 1, 0, 2, 1, 2, 0], dtype=torch.int64, device=hex.device)
        B[:, idx_B[0], idx_B[1], :] = inv_N[:, idx_inv_N, :]

        B = B.transpose(2, 3).contiguous().reshape(27, 6, 24)

        # Gaussian integration
        # k=B^T @ C @ B *w

        stiffness = (B.transpose(1, 2)@elastic_tensor@B*w).sum(0)
        force = (B.transpose(1, 2)@elastic_tensor*w).sum(0)
        return stiffness, force

    def macro_deformation(self, lx, ly, lz, elastic_tensor):
        hex = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [
                           1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=elastic_tensor.dtype, device=self.device)
        scale = torch.tensor(
            [lx, ly, lz], dtype=elastic_tensor.dtype, device=self.device)
        hex = torch.div(hex, scale)

        idx = torch.ones(24, dtype=torch.bool, device=self.device)
        idx[[0, 1, 2, 4, 5, 11]] = False

        ke, fe = self.stiffness_force(hex, elastic_tensor)

        X0 = torch.zeros(24, 6, dtype=elastic_tensor.dtype, device=self.device)
        X0[idx, :] = torch.inverse(ke[idx, :][:, idx])@fe[idx, :]

        return ke, X0


def isotropic_elastic_tensor(E, v):
    Lambda = v / (1. + v) / (1 - 2. * v)*E
    Mu = 1. / (2.*(1. + v))*E

    return torch.as_tensor([
        [Lambda + 2 * Mu, Lambda, Lambda, 0, 0, 0],
        [Lambda, Lambda + 2 * Mu, Lambda, 0, 0, 0],
        [Lambda, Lambda, Lambda + 2 * Mu, 0, 0, 0],
        [0, 0, 0, Mu, 0, 0],
        [0, 0, 0, 0, Mu, 0],
        [0, 0, 0, 0, 0, Mu]])


def shape_material_transform(material, J):
    if J.dim() == 2:
        MA = torch.tensor([[torch.square(J[0, 0]), torch.square(J[0, 1]), torch.square(J[0, 2]),
                            J[0, 0] * J[0, 1], J[0, 1] * J[0, 2], J[0, 0] * J[0, 2]],
                           [torch.square(J[1, 0]), torch.square(J[1, 1]), torch.square(J[1, 2]),
                            J[1, 0] * J[1, 1], J[1, 1] * J[1, 2], J[1, 0] * J[1, 2]],
                           [torch.square(J[2, 0]), torch.square(J[2, 1]), torch.square(J[2, 2]),
                            J[2, 0] * J[2, 1], J[2, 1] * J[2, 2], J[2, 0] * J[2, 2]],
                           [2 * J[0, 0] * J[1, 0], 2 * J[0, 1] * J[1, 1],
                            2 * J[0, 2] * J[1, 2], J[0, 0] *
                            J[1, 1] + J[0, 1] * J[1, 0],
                            J[0, 1] * J[1, 2] + J[0, 2] * J[1, 1],
                            J[0, 0] * J[1, 2] + J[0, 2] * J[1, 0]],
                           [2 * J[1, 0] * J[2, 0], 2 * J[1, 1] * J[2, 1],
                            2 * J[1, 2] * J[2, 2], J[1, 0] *
                            J[2, 1] + J[1, 1] * J[2, 0],
                            J[1, 1] * J[2, 2] + J[1, 2] * J[2, 1],
                            J[1, 0] * J[2, 2] + J[1, 2] * J[2, 0]],
                           [2 * J[2, 0] * J[0, 0], 2 * J[2, 1] * J[0, 1],
                            2 * J[2, 2] * J[0, 2], J[2, 0] *
                            J[0, 1] + J[2, 1] * J[0, 0],
                            J[2, 1] * J[0, 2] + J[2, 2] * J[0, 1],
                            J[2, 0] * J[0, 2] + J[2, 2] * J[0, 0]],
                           ], dtype=material.dtype, device=material.device)
        return MA.t() @ material @ MA

    elif J.dim() == 3:
        MA = torch.empty(J.shape[0], 6, 6,
                         dtype=material.dtype, device=material.device)
        MA[:, 0, :] = torch.vstack([torch.square(J[:, 0, 0]), torch.square(J[:, 0, 1]), torch.square(
            J[:, 0, 2]), J[:, 0, 0] * J[:, 0, 1], J[:, 0, 1] * J[:, 0, 2], J[:, 0, 0] * J[:, 0, 2]]).t()
        MA[:, 1, :] = torch.vstack([torch.square(J[:, 1, 0]), torch.square(J[:, 1, 1]), torch.square(
            J[:, 1, 2]), J[:, 1, 0] * J[:, 1, 1], J[:, 1, 1] * J[:, 1, 2], J[:, 1, 0] * J[:, 1, 2]]).t()
        MA[:, 2, :] = torch.vstack([torch.square(J[:, 2, 0]), torch.square(J[:, 2, 1]), torch.square(
            J[:, 2, 2]), J[:, 2, 0] * J[:, 2, 1], J[:, 2, 1] * J[:, 2, 2], J[:, 2, 0] * J[:, 2, 2]]).t()
        MA[:, 3, :] = torch.vstack([2 * J[:, 0, 0] * J[:, 1, 0], 2 * J[:, 0, 1] * J[:, 1, 1], 2 * J[:, 0, 2] * J[:, 1, 2], J[:, 0, 0] * J[:, 1, 1] +
                                   J[:, 0, 1] * J[:, 1, 0], J[:, 0, 1] * J[:, 1, 2] + J[:, 0, 2] * J[:, 1, 1], J[:, 0, 0] * J[:, 1, 2] + J[:, 0, 2] * J[:, 1, 0]]).t()
        MA[:, 4, :] = torch.vstack([2 * J[:, 1, 0] * J[:, 2, 0], 2 * J[:, 1, 1] * J[:, 2, 1], 2 * J[:, 1, 2] * J[:, 2, 2], J[:, 1, 0] * J[:, 2, 1] +
                                   J[:, 1, 1] * J[:, 2, 0], J[:, 1, 1] * J[:, 2, 2] + J[:, 1, 2] * J[:, 2, 1], J[:, 1, 0] * J[:, 2, 2] + J[:, 1, 2] * J[:, 2, 0]]).t()
        MA[:, 5, :] = torch.vstack([2 * J[:, 2, 0] * J[:, 0, 0], 2 * J[:, 2, 1] * J[:, 0, 1], 2 * J[:, 2, 2] * J[:, 0, 2], J[:, 2, 0] * J[:, 0, 1] +
                                   J[:, 2, 1] * J[:, 0, 0], J[:, 2, 1] * J[:, 0, 2] + J[:, 2, 2] * J[:, 0, 1], J[:, 2, 0] * J[:, 0, 2] + J[:, 2, 2] * J[:, 0, 0]]).t()

        return MA.transpose(1, 2) @ material @ MA

def cell_dof_indices(nelx, nely, nelz):
    nodeidx = torch.zeros(1+nelx, 1+nely, 1+nelz,
                            dtype=torch.long)
    nodeidx[:-1, :-1, :-1] = torch.arange(
        nelx*nely*nelz).reshape(nelx, nely, nelz)
    nodeidx[-1, :, :] = nodeidx[0, :, :]
    nodeidx[:, -1, :] = nodeidx[:, 0, :]
    nodeidx[:, :, -1] = nodeidx[:, :, 0]

    cellidx = torch.empty(
        nelx*nely*nelz, 8, dtype=torch.long)
    cellidx[:, 0] = nodeidx[:-1, :-1, :-1].reshape(-1)
    cellidx[:, 1] = nodeidx[1:, :-1, :-1].reshape(-1)
    cellidx[:, 2] = nodeidx[1:, 1:, :-1].reshape(-1)
    cellidx[:, 3] = nodeidx[:-1, 1:, :-1].reshape(-1)
    cellidx[:, 4] = nodeidx[:-1, :-1, 1:].reshape(-1)
    cellidx[:, 5] = nodeidx[1:, :-1, 1:].reshape(-1)
    cellidx[:, 6] = nodeidx[1:, 1:, 1:].reshape(-1)
    cellidx[:, 7] = nodeidx[:-1, 1:, 1:].reshape(-1)

    offset = torch.tensor([0, 1, 2], dtype=torch.long).repeat(8)
    return torch.repeat_interleave(cellidx, 3, dim=1)*3+offset

