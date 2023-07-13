import torch


def eval(homo_net,property, voxel, output, Ke_hard, Fe_hard, ke, X0):
    """computer batch error between predicated homogenized tensor and the groundtruth

    Args:
        H (object): Homo 3D full instance
        H1 (object): Homo 3D instance
        C (torch.cuda.FloatTensor): base material elastic tensor, it should be batch_size*36
        voxel (torch.cuda.FloatTensor):  it should be batch_size*36*N*N*N
        output (torch.cuda.FloatTensor): predicated displacement,it should be batch_size*18*N*N*N
        transform (torch.cuda.FloatTensor): Jacobian transformer, it should be batch_size*3*3
        Ke (torch.cuda.FloatTensor): element stiffness matrics, it should be batch_size*24*24
        Fe (torch.cuda.FloatTensor): element force matrics, it should be batch_size*24*6

    Returns:
        float: all_error
        torch.cuda.FloatTensor: batch_error, it should be batch_size
    """
    batch_size = voxel.shape[0]
    batch_predict = torch.zeros(
        batch_size, 6, 6, dtype=Ke_hard.dtype, device=homo_net.device)
    batch_actual = property
    batch_error = torch.zeros(
        batch_size, dtype=Ke_hard.dtype, device=homo_net.device)
    for i in range(batch_size):

        C_homo = homo_net.homogenized(voxel[i], output[i], ke, X0)
        # C_final=tensor_transform(C_homo,jacobi[i])
        batch_predict[i] = C_homo

        batch_error[i] = torch.dist(
            batch_actual[i], batch_predict[i])/torch.linalg.norm(batch_actual[i])

    return batch_error.mean(), batch_error
